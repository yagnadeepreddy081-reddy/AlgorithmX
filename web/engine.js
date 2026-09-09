/* ===========================================================================
   SAMUDRA NETRA -- in-browser inference engine.

   This runs the *actual trained U-Net* in the page. No server, no API, no
   network: the weights are baked into the file as float16 and the forward pass
   is plain JavaScript over Float32Arrays. Consequences that matter for a demo:
   a judge can hand over any image, it is analysed on the spot, and it still
   works with the Wi-Fi switched off.

   BatchNorm was folded into the convolutions at export time, so every block
   here is convolution -> bias -> ReLU.

   Layout: tensors are {c, h, w, d} with d a Float32Array in CHW order.
   =========================================================================== */
var SN = (function () {
  "use strict";

  /* ---------- float16 -> float32 -------------------------------------- */
  function f16to32(u) {
    var s = (u & 0x8000) >> 15, e = (u & 0x7c00) >> 10, f = u & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function loadWeights(spec) {
    var bin = atob(spec.data);
    var n = spec.count;
    var u16 = new Uint16Array(n);
    for (var i = 0; i < n; i++) u16[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    var f32 = new Float32Array(n);
    for (i = 0; i < n; i++) f32[i] = f16to32(u16[i]);
    var W = {};
    spec.tensors.forEach(function (t) {
      W[t.name] = { shape: t.shape, d: f32.subarray(t.offset, t.offset + t.size) };
    });
    return W;
  }

  /* ---------- tensor helpers ------------------------------------------ */
  function tensor(c, h, w) { return { c: c, h: h, w: w, d: new Float32Array(c * h * w) }; }

  /* 3x3 convolution, zero padded, bias added, optional ReLU.

     The input plane is copied once into a (H+2)x(W+2) buffer so the inner loop
     carries no bounds checks -- that single change roughly doubles throughput
     in V8, which is the difference between a four second wait and a ten. */
  function conv3(x, w, b, cout, relu) {
    var Cin = x.c, H = x.h, W = x.w, HW = H * W;
    var out = tensor(cout, H, W), od = out.d, xd = x.d, wd = w.d, bd = b.d;
    var oc, ic, y, xx, o, p0, p1, p2;
    for (oc = 0; oc < cout; oc++) {
      var bias = bd[oc], base = oc * HW;
      for (o = 0; o < HW; o++) od[base + o] = bias;
    }
    var PW = W + 2;
    var P = new Float32Array((H + 2) * PW);
    for (ic = 0; ic < Cin; ic++) {
      P.fill(0);
      var src = ic * HW;
      for (y = 0; y < H; y++) {
        var s = src + y * W, dst = (y + 1) * PW + 1;
        for (xx = 0; xx < W; xx++) P[dst + xx] = xd[s + xx];
      }
      for (oc = 0; oc < cout; oc++) {
        var k = ((oc * Cin) + ic) * 9;
        var k0 = wd[k], k1 = wd[k + 1], k2 = wd[k + 2],
            k3 = wd[k + 3], k4 = wd[k + 4], k5 = wd[k + 5],
            k6 = wd[k + 6], k7 = wd[k + 7], k8 = wd[k + 8];
        if (k0 === 0 && k1 === 0 && k2 === 0 && k3 === 0 && k4 === 0 &&
            k5 === 0 && k6 === 0 && k7 === 0 && k8 === 0) continue;
        var ob = oc * HW;
        for (y = 0; y < H; y++) {
          p0 = y * PW; p1 = p0 + PW; p2 = p1 + PW;
          o = ob + y * W;
          // sliding 3-tap window: two of every three taps are already in
          // registers from the previous column, so each output pixel costs
          // three loads from P instead of nine
          var a0 = P[p0], a1 = P[p0 + 1],
              b0 = P[p1], b1 = P[p1 + 1],
              c0 = P[p2], c1 = P[p2 + 1], a2, b2, c2;
          for (xx = 0; xx < W; xx++) {
            a2 = P[p0 + xx + 2]; b2 = P[p1 + xx + 2]; c2 = P[p2 + xx + 2];
            od[o + xx] += k0 * a0 + k1 * a1 + k2 * a2 +
                          k3 * b0 + k4 * b1 + k5 * b2 +
                          k6 * c0 + k7 * c1 + k8 * c2;
            a0 = a1; a1 = a2; b0 = b1; b1 = b2; c0 = c1; c1 = c2;
          }
        }
      }
    }
    if (relu) for (var i = 0; i < od.length; i++) if (od[i] < 0) od[i] = 0;
    return out;
  }

  function conv1(x, w, b, cout) {
    var Cin = x.c, HW = x.h * x.w;
    var out = tensor(cout, x.h, x.w), od = out.d, xd = x.d, wd = w.d, bd = b.d;
    for (var oc = 0; oc < cout; oc++) {
      var ob = oc * HW, bias = bd[oc];
      for (var i = 0; i < HW; i++) od[ob + i] = bias;
      for (var ic = 0; ic < Cin; ic++) {
        var kv = wd[oc * Cin + ic];
        if (kv === 0) continue;
        var ib = ic * HW;
        for (i = 0; i < HW; i++) od[ob + i] += kv * xd[ib + i];
      }
    }
    return out;
  }

  function maxpool2(x) {
    var H = x.h >> 1, W = x.w >> 1;
    var out = tensor(x.c, H, W), od = out.d, xd = x.d;
    for (var c = 0; c < x.c; c++) {
      var sb = c * x.h * x.w, db = c * H * W;
      for (var y = 0; y < H; y++) {
        var r0 = sb + (2 * y) * x.w, r1 = r0 + x.w, o = db + y * W;
        for (var i = 0; i < W; i++) {
          var a = xd[r0 + 2 * i], b = xd[r0 + 2 * i + 1],
              c2 = xd[r1 + 2 * i], d2 = xd[r1 + 2 * i + 1];
          var m = a > b ? a : b; if (c2 > m) m = c2; if (d2 > m) m = d2;
          od[o + i] = m;
        }
      }
    }
    return out;
  }

  /* ConvTranspose2d, kernel 2, stride 2 -- each input pixel writes a disjoint
     2x2 output block, so no accumulation across neighbours is needed. */
  function upconv2(x, w, b, cout) {
    var Cin = x.c, H = x.h, W = x.w, OH = H * 2, OW = W * 2;
    var out = tensor(cout, OH, OW), od = out.d, xd = x.d, wd = w.d, bd = b.d;
    var OHW = OH * OW;
    for (var oc = 0; oc < cout; oc++) {
      var ob = oc * OHW, bias = bd[oc];
      for (var i = 0; i < OHW; i++) od[ob + i] = bias;
    }
    for (var ic = 0; ic < Cin; ic++) {
      var ib = ic * H * W;
      for (oc = 0; oc < cout; oc++) {
        var k = ((ic * cout) + oc) * 4;   // torch layout: (in, out, kh, kw)
        var k0 = wd[k], k1 = wd[k + 1], k2 = wd[k + 2], k3 = wd[k + 3];
        var ob2 = oc * OHW;
        for (var y = 0; y < H; y++) {
          var s = ib + y * W, r0 = ob2 + (2 * y) * OW, r1 = r0 + OW;
          for (var xx = 0; xx < W; xx++) {
            var v = xd[s + xx];
            if (v === 0) continue;
            var o0 = r0 + 2 * xx, o1 = r1 + 2 * xx;
            od[o0] += k0 * v; od[o0 + 1] += k1 * v;
            od[o1] += k2 * v; od[o1 + 1] += k3 * v;
          }
        }
      }
    }
    return out;
  }

  function cat(a, b) {
    var out = tensor(a.c + b.c, a.h, a.w);
    out.d.set(a.d, 0);
    out.d.set(b.d, a.d.length);
    return out;
  }

  function argmaxSoftmax(x) {
    var C = x.c, HW = x.h * x.w, d = x.d;
    var cls = new Uint8Array(HW), probs = new Float32Array(C * HW);
    for (var i = 0; i < HW; i++) {
      var m = -Infinity, mi = 0, c;
      for (c = 0; c < C; c++) { var v = d[c * HW + i]; if (v > m) { m = v; mi = c; } }
      var s = 0;
      for (c = 0; c < C; c++) { var e = Math.exp(d[c * HW + i] - m); probs[c * HW + i] = e; s += e; }
      for (c = 0; c < C; c++) probs[c * HW + i] /= s;
      cls[i] = mi;
    }
    return { cls: cls, probs: probs, h: x.h, w: x.w, c: C };
  }

  function tick() { return new Promise(function (r) { setTimeout(r, 0); }); }

  /* ---------- forward pass -------------------------------------------- */
  async function forward(W, imgGray, size, onProgress) {
    var steps = 15, done = 0;
    function step(label) {
      done++;
      if (onProgress) onProgress(done / steps, label);
      return tick();
    }
    var x = { c: 1, h: size, w: size, d: imgGray };

    var e1 = conv3(x, W["enc1.c1.w"], W["enc1.c1.b"], 20, true); await step("encoder 1");
    e1 = conv3(e1, W["enc1.c2.w"], W["enc1.c2.b"], 20, true); await step("encoder 1");

    var p1 = maxpool2(e1);
    var e2 = conv3(p1, W["enc2.c1.w"], W["enc2.c1.b"], 40, true); await step("encoder 2");
    e2 = conv3(e2, W["enc2.c2.w"], W["enc2.c2.b"], 40, true); await step("encoder 2");

    var p2 = maxpool2(e2);
    var e3 = conv3(p2, W["enc3.c1.w"], W["enc3.c1.b"], 80, true); await step("encoder 3");
    e3 = conv3(e3, W["enc3.c2.w"], W["enc3.c2.b"], 80, true); await step("encoder 3");

    var p3 = maxpool2(e3);
    var bt = conv3(p3, W["bott.c1.w"], W["bott.c1.b"], 160, true); await step("bottleneck");
    bt = conv3(bt, W["bott.c2.w"], W["bott.c2.b"], 160, true); await step("bottleneck");

    var u3 = upconv2(bt, W["up3.w"], W["up3.b"], 80);
    var d3 = conv3(cat(u3, e3), W["dec3.c1.w"], W["dec3.c1.b"], 80, true); await step("decoder 3");
    d3 = conv3(d3, W["dec3.c2.w"], W["dec3.c2.b"], 80, true); await step("decoder 3");

    var u2 = upconv2(d3, W["up2.w"], W["up2.b"], 40);
    var d2 = conv3(cat(u2, e2), W["dec2.c1.w"], W["dec2.c1.b"], 40, true); await step("decoder 2");
    d2 = conv3(d2, W["dec2.c2.w"], W["dec2.c2.b"], 40, true); await step("decoder 2");

    var u1 = upconv2(d2, W["up1.w"], W["up1.b"], 20);
    var d1 = conv3(cat(u1, e1), W["dec1.c1.w"], W["dec1.c1.b"], 20, true); await step("decoder 1");
    d1 = conv3(d1, W["dec1.c2.w"], W["dec1.c2.b"], 20, true); await step("decoder 1");

    var logits = conv1(d1, W["head.w"], W["head.b"], 5); await step("classifying pixels");
    return argmaxSoftmax(logits);
  }

  /* =====================================================================
     Image intake.

     A judge may hand over anything: a Sentinel-1 quicklook, a screenshot, a
     holiday photo. The plausibility check below reports what the image
     actually is rather than silently pretending a beach photo is radar.
     ===================================================================== */
  function prepare(imageEl, size) {
    var cv = document.createElement("canvas");
    cv.width = size; cv.height = size;
    var ctx = cv.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(imageEl, 0, 0, size, size);
    var px = ctx.getImageData(0, 0, size, size).data;
    var g = new Float32Array(size * size);
    var satSum = 0, n = size * size;
    for (var i = 0; i < n; i++) {
      var r = px[4 * i], gg = px[4 * i + 1], b = px[4 * i + 2];
      var mx = Math.max(r, gg, b), mn = Math.min(r, gg, b);
      satSum += mx > 0 ? (mx - mn) / mx : 0;
      g[i] = (0.299 * r + 0.587 * gg + 0.114 * b) / 255;
    }
    // statistics used by the plausibility check
    var mean = 0; for (i = 0; i < n; i++) mean += g[i]; mean /= n;
    var vr = 0; for (i = 0; i < n; i++) { var dv = g[i] - mean; vr += dv * dv; } vr /= n;
    var sd = Math.sqrt(vr);
    // local speckle: mean absolute difference between horizontal neighbours
    var mad = 0, cnt = 0;
    for (var y = 0; y < size; y++)
      for (var x = 0; x < size - 1; x++) { mad += Math.abs(g[y * size + x] - g[y * size + x + 1]); cnt++; }
    mad /= cnt;
    var sat = satSum / n;

    var notes = [], ok = true;
    if (sat > 0.16) {
      ok = false;
      notes.push("Strongly coloured image. Sentinel-1 GRD is single-channel radar backscatter and has no colour; converted to greyscale and analysed anyway, but treat the result as a demonstration of the pipeline, not a detection.");
    }
    if (mad < 0.035) {
      ok = false;
      notes.push("Pixel-to-pixel variation is far lower than SAR speckle. Real Sentinel-1 GRD at 4.4 looks has a characteristic grain this image does not show, which usually means it is an optical photograph or a heavily smoothed graphic.");
    }
    if (sd < 0.035) {
      ok = false;
      notes.push("Almost no contrast across the tile.");
    }
    if (ok) notes.push("Speckle statistics and dynamic range are consistent with a Sentinel-1 GRD quicklook.");
    return {
      gray: g, size: size, canvas: cv,
      stats: { mean: mean, sd: sd, speckle: mad, saturation: sat },
      plausible: ok, notes: notes
    };
  }

  return {
    loadWeights: loadWeights,
    forward: forward,
    prepare: prepare,
    tick: tick
  };
})();
