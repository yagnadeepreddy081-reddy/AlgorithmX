/* ===========================================================================
   SAMUDRA NETRA -- live analysis of an uploaded scene.

   Everything the Python pipeline does for stages 2 to 4 is reimplemented here
   so it can run on an image a judge hands over during the demo:

     stage 2  segmentation            (engine.js, the real trained network)
     stage 3  fragment grouping, shape measurement, look-alike discrimination
     stage 4  backward Lagrangian drift ensemble -> release corridor

   Stages 5 and 6 -- AIS correlation and vessel scoring -- are deliberately NOT
   run on an uploaded scene, and the interface says so rather than inventing a
   ship. Attribution needs the vessel-traffic archive for that scene's date and
   position. We hold that archive only for the pilot box, so the honest place to
   demonstrate the full chain is the 38-scene test set, where the AIS tracks and
   the ground truth both exist.

   The numbers below are the same constants as the Python: PIXEL_METRES 80,
   wind validity band 3-12 m/s, damping band 3-10 dB, wind-drift factor
   0.020-0.045, eddy diffusivity 25 m^2/s, 12 km usability threshold.
   =========================================================================== */
var SNA = (function () {
  "use strict";

  var PIXEL_M = 80.0;
  var MIN_AREA_PX = 60;
  var GROUP_DILATE = 8;
  var MAX_USEFUL_KM = 12.0;

  /* ---------- morphology --------------------------------------------- */
  function dilate(mask, n, S) {
    var cur = mask, i, k, y, x;
    for (k = 0; k < n; k++) {
      var out = new Uint8Array(S * S);
      for (y = 0; y < S; y++) {
        for (x = 0; x < S; x++) {
          if (cur[y * S + x]) { out[y * S + x] = 1; continue; }
          var hit = 0;
          for (var dy = -1; dy <= 1 && !hit; dy++) {
            var yy = y + dy; if (yy < 0 || yy >= S) continue;
            for (var dx = -1; dx <= 1; dx++) {
              var xx = x + dx; if (xx < 0 || xx >= S) continue;
              if (cur[yy * S + xx]) { hit = 1; break; }
            }
          }
          out[y * S + x] = hit;
        }
      }
      cur = out;
    }
    return cur;
  }

  function erode(mask, S) {
    var out = new Uint8Array(S * S);
    for (var y = 1; y < S - 1; y++)
      for (var x = 1; x < S - 1; x++) {
        var i = y * S + x;
        if (mask[i] && mask[i - 1] && mask[i + 1] && mask[i - S] && mask[i + S]) out[i] = 1;
      }
    return out;
  }

  /* Flood-fill labelling of the dilated mask, then intersect back -- the same
     grouping the Python does, and for the same reason: a discharge appears as
     a chain of windrows, not one continuous film. */
  function groupRegions(mask, S) {
    var grown = dilate(mask, GROUP_DILATE, S);
    var lab = new Int32Array(S * S).fill(-1);
    var regions = [], stack = new Int32Array(S * S), n = 0;
    for (var i = 0; i < S * S; i++) {
      if (!grown[i] || lab[i] >= 0) continue;
      var id = regions.length, sp = 0;
      stack[sp++] = i; lab[i] = id;
      var members = [];
      while (sp > 0) {
        var p = stack[--sp];
        if (mask[p]) members.push(p);
        var py = (p / S) | 0, px = p % S;
        for (var dy = -1; dy <= 1; dy++) {
          var yy = py + dy; if (yy < 0 || yy >= S) continue;
          for (var dx = -1; dx <= 1; dx++) {
            var xx = px + dx; if (xx < 0 || xx >= S) continue;
            var q = yy * S + xx;
            if (grown[q] && lab[q] < 0) { lab[q] = id; stack[sp++] = q; }
          }
        }
      }
      if (members.length >= MIN_AREA_PX) regions.push(members);
      else regions.push(null);
      n++;
    }
    return regions.filter(function (r) { return r && r.length >= MIN_AREA_PX; });
  }

  function median(arr) {
    if (!arr.length) return 0;
    var a = Array.prototype.slice.call(arr).sort(function (p, q) { return p - q; });
    return a[a.length >> 1];
  }

  /* ---------- shape measurement --------------------------------------- */
  function measure(members, gray, mask, probs, S) {
    var n = members.length, i;
    var sy = 0, sx = 0;
    for (i = 0; i < n; i++) { sy += (members[i] / S) | 0; sx += members[i] % S; }
    var cy = sy / n, cx = sx / n;

    var vxx = 0, vyy = 0, vxy = 0;
    for (i = 0; i < n; i++) {
      var dy = ((members[i] / S) | 0) - cy, dx = (members[i] % S) - cx;
      vxx += dx * dx; vyy += dy * dy; vxy += dx * dy;
    }
    vxx /= n; vyy /= n; vxy /= n;
    var tr = vxx + vyy, det = vxx * vyy - vxy * vxy;
    var disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
    var l1 = tr / 2 + disc, l2 = Math.max(tr / 2 - disc, 1e-9);
    // principal axis; image y grows southwards so the north component is -vy
    var vx = vxy, vy = l1 - vxx;
    if (Math.abs(vxy) < 1e-9) { vx = vxx >= vyy ? 1 : 0; vy = vxx >= vyy ? 0 : 1; }
    var norm = Math.hypot(vx, vy) || 1; vx /= norm; vy /= norm;
    var orient = ((Math.atan2(vx, -vy) * 180 / Math.PI) % 180 + 180) % 180;
    var elong = Math.sqrt(l1 / l2);

    var ux = Math.sin(orient * Math.PI / 180), uy = -Math.cos(orient * Math.PI / 180);
    var pmin = Infinity, pmax = -Infinity, qmin = Infinity, qmax = -Infinity;
    for (i = 0; i < n; i++) {
      var ddy = ((members[i] / S) | 0) - cy, ddx = (members[i] % S) - cx;
      var p = ddx * ux + ddy * uy, q = -ddx * uy + ddy * ux;
      if (p < pmin) pmin = p; if (p > pmax) pmax = p;
      if (q < qmin) qmin = q; if (q > qmax) qmax = q;
    }

    // contrast: eroded core against a ring of clean sea a few pixels out
    var rmask = new Uint8Array(S * S);
    for (i = 0; i < n; i++) rmask[members[i]] = 1;
    var core = erode(rmask, S);
    var coreVals = [], anyCore = false;
    for (i = 0; i < S * S; i++) if (core[i]) { coreVals.push(gray[i]); anyCore = true; }
    if (!anyCore) for (i = 0; i < n; i++) coreVals.push(gray[members[i]]);
    var d6 = dilate(rmask, 6, S), d2 = dilate(rmask, 2, S);
    var ringVals = [];
    for (i = 0; i < S * S; i++) if (d6[i] && !d2[i] && mask[i] === 0) ringVals.push(gray[i]);
    var inside = median(coreVals);
    var surround = ringVals.length > 20 ? median(ringVals) : median(gray);
    var contrastDb = (surround - inside) * 27.0;   // display stretch spans 27 dB

    // edge sharpness: mean gradient magnitude on the region boundary
    var er = erode(rmask, S), gsum = 0, gc = 0;
    for (var y = 1; y < S - 1; y++)
      for (var x = 1; x < S - 1; x++) {
        i = y * S + x;
        if (!rmask[i] || er[i]) continue;
        var gyv = (gray[i + S] - gray[i - S]) / 2, gxv = (gray[i + 1] - gray[i - 1]) / 2;
        gsum += Math.hypot(gxv, gyv); gc++;
      }
    var sharp = gc ? gsum / gc : 0;

    var HW = S * S, pOil = 0, pLa = 0;
    for (i = 0; i < n; i++) { pOil += probs[1 * HW + members[i]]; pLa += probs[2 * HW + members[i]]; }
    pOil /= n; pLa /= n;

    return {
      npx: n, cy: cy, cx: cx,
      area_km2: n * PIXEL_M * PIXEL_M / 1e6,
      length_km: (pmax - pmin) * PIXEL_M / 1000,
      width_km: (qmax - qmin) * PIXEL_M / 1000,
      elongation: elong,
      orientation_deg: orient,
      damping_contrast_db: contrastDb,
      edge_sharpness: sharp,
      model_oil_prob: pOil,
      model_lookalike_prob: pLa,
      members: members
    };
  }

  function windValidity(w) {
    if (w <= 2.5 || w >= 14.0) return 0.10;
    if (w < 4.0) return 0.10 + 0.90 * (w - 2.5) / 1.5;
    if (w > 11.0) return 0.10 + 0.90 * (14.0 - w) / 3.0;
    return 1.0;
  }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

  function discriminate(m, wind) {
    var wv = windValidity(wind);
    var elongS = clamp01((m.elongation - 1.6) / 3.4);
    var sharpS = clamp01(m.edge_sharpness / 0.06);
    var dampS = clamp01((m.damping_contrast_db - 1.2) / 1.8) * clamp01((12.0 - m.damping_contrast_db) / 3.0);
    var modelS = clamp01(m.model_oil_prob / Math.max(m.model_oil_prob + m.model_lookalike_prob, 1e-6));
    var conf = wv * (0.40 * modelS + 0.22 * elongS + 0.20 * dampS + 0.18 * sharpS);
    var verdict = conf >= 0.55 ? "oil (high confidence)"
                : conf >= 0.35 ? "probable oil - analyst review"
                : "likely look-alike - not escalated";
    return {
      confidence: conf, verdict: verdict, wind_ms: wind, wind_validity: wv,
      factors: {
        "model evidence": modelS,
        "elongation": elongS,
        "damping in 3-10 dB band": dampS,
        "edge sharpness": sharpS,
        "wind validity gate": wv
      }
    };
  }

  /* ---------- environmental field (same model as fields.py) ------------ */
  function Field(seed, meanWind, windDir) {
    var s = seed >>> 0;
    function rnd() { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }
    this.meanWind = meanWind; this.windDir = windDir;
    this.mu = -0.13 + (rnd() - 0.5) * 0.08;
    this.mv = 0.22 + (rnd() - 0.5) * 0.10;
    this.phase = rnd() * 2 * Math.PI;
    this.eddies = [];
    for (var i = 0; i < 2; i++)
      this.eddies.push({ x: (rnd() - 0.5) * 50000, y: (rnd() - 0.5) * 50000,
                         r: 9000 + rnd() * 11000, amp: (rnd() - 0.5) * 0.44 });
  }
  Field.prototype.wind = function (t) {
    var th = this.windDir * Math.PI / 180;
    var u = -this.meanWind * Math.sin(th), v = -this.meanWind * Math.cos(th);
    var veer = (6.0 * Math.PI / 180) * (t / 86400);
    return [u * Math.cos(veer) - v * Math.sin(veer), u * Math.sin(veer) + v * Math.cos(veer)];
  };
  Field.prototype.current = function (x, y, t) {
    var u = this.mu, v = this.mv;
    var ph = 2 * Math.PI * t / (12.42 * 3600) + this.phase;
    u += 0.16 * Math.cos(ph); v += 0.09 * Math.sin(ph);
    for (var i = 0; i < this.eddies.length; i++) {
      var e = this.eddies[i], dx = x - e.x, dy = y - e.y, d = Math.hypot(dx, dy);
      if (d < 1) continue;
      var g = Math.exp(-(d * d) / (2 * e.r * e.r));
      u += e.amp * g * (-dy / d); v += e.amp * g * (dx / d);
    }
    return [u, v];
  };

  function gauss(rndF) {
    var u = Math.max(rndF(), 1e-9), v = rndF();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---------- backward drift ensemble ---------------------------------- */
  function backwardDrift(m, S, wind, windDir, hoursBack, seed) {
    var N = 900, M = 12, DT = 900, KH = 25.0;
    var s = (seed || 12345) >>> 0;
    function rnd() { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }
    var field = new Field(seed || 7, wind, windDir);

    var x = new Float64Array(N), y = new Float64Array(N);
    var alpha = new Float64Array(N), defl = new Float64Array(N),
        cg = new Float64Array(N), wg = new Float64Array(N), crot = new Float64Array(N);
    var mem = [];
    for (var k = 0; k < M; k++)
      mem.push({ a: 0.020 + rnd() * 0.025, d: (-10 + rnd() * 45) * Math.PI / 180,
                 c: Math.max(0.25, 1 + gauss(rnd) * 0.30),
                 w: Math.max(0.25, 1 + gauss(rnd) * 0.22),
                 r: gauss(rnd) * 20 * Math.PI / 180 });
    for (var i = 0; i < N; i++) {
      var p = m.members[(rnd() * m.members.length) | 0];
      var py = ((p / S) | 0) + rnd() - 0.5, px = (p % S) + rnd() - 0.5;
      x[i] = (px - S / 2) * PIXEL_M; y[i] = (S / 2 - py) * PIXEL_M;
      var mm = mem[i % M];
      alpha[i] = mm.a; defl[i] = mm.d; cg[i] = mm.c; wg[i] = mm.w; crot[i] = mm.r;
    }

    var sig = Math.sqrt(2 * KH * DT), frames = [], t = 0, r90zero = 0;
    for (var h = 0; h <= hoursBack; h++) {
      if (h > 0) {
        for (var st = 0; st < 4; st++) {
          for (i = 0; i < N; i++) {
            function velocity(px, py, at) {
              var c = field.current(px, py, at), w = field.wind(at);
              var ca = Math.cos(defl[i]), sa = Math.sin(defl[i]);
              var cr = Math.cos(crot[i]), sr = Math.sin(crot[i]);
              return [cg[i]*(c[0]*cr-c[1]*sr) + alpha[i]*wg[i]*(w[0]*ca+w[1]*sa),
                      cg[i]*(c[0]*sr+c[1]*cr) + alpha[i]*wg[i]*(-w[0]*sa+w[1]*ca)];
            }
            var vel = velocity(x[i], y[i], t);
            var mid = velocity(x[i]-vel[0]*DT/2, y[i]-vel[1]*DT/2, t-DT/2);
            x[i] -= mid[0]*DT; y[i] -= mid[1]*DT;
            x[i] += gauss(rnd) * sig; y[i] += gauss(rnd) * sig;
          }
          t -= DT;
        }
      }
      var mx = 0, my = 0;
      for (i = 0; i < N; i++) { mx += x[i]; my += y[i]; }
      mx /= N; my /= N;
      var dd = new Float64Array(N);
      for (i = 0; i < N; i++) dd[i] = Math.hypot(x[i] - mx, y[i] - my);
      var sorted = Array.prototype.slice.call(dd).sort(function (a, b) { return a - b; });
      var r90 = sorted[(0.9 * (N - 1)) | 0] / 1000;
      if (h === 0) r90zero = r90;
      var added = Math.max(0, r90 - r90zero);
      var pts = [];
      for (i = 0; i < N; i += 4) pts.push([x[i], y[i]]);
      frames.push({ hours_back: h, cx: mx, cy: my, r90_km: r90,
                    uncertainty_km: added, usable: added <= MAX_USEFUL_KM, pts: pts });
    }
    var usable = frames.filter(function (f) { return f.usable; });
    return { frames: frames, hours_computed: hoursBack,
             hours_usable: usable.length ? usable.length - 1 : 0,
             particles: N, ensemble_members: M, max_useful_radius_km: MAX_USEFUL_KM };
  }

  /* ---------- rendering ------------------------------------------------ */
  var COLOURS = { 1: [255, 62, 62], 2: [255, 208, 61], 3: [78, 255, 154], 4: [150, 122, 84] };

  function toDataURLs(gray, cls, S) {
    var cv = document.createElement("canvas"); cv.width = S; cv.height = S;
    var ctx = cv.getContext("2d");
    var im = ctx.createImageData(S, S), d = im.data;
    for (var i = 0; i < S * S; i++) {
      var g = Math.max(0, Math.min(255, Math.round(gray[i] * 255)));
      d[4 * i] = g; d[4 * i + 1] = g; d[4 * i + 2] = g; d[4 * i + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    var sar = cv.toDataURL("image/png");

    for (i = 0; i < S * S; i++) {
      var c = COLOURS[cls[i]];
      if (!c) continue;
      d[4 * i] = Math.round(d[4 * i] * 0.45 + c[0] * 0.55);
      d[4 * i + 1] = Math.round(d[4 * i + 1] * 0.45 + c[1] * 0.55);
      d[4 * i + 2] = Math.round(d[4 * i + 2] * 0.45 + c[2] * 0.55);
    }
    ctx.putImageData(im, 0, 0);
    return { sar: sar, overlay: cv.toDataURL("image/png") };
  }

  /* ---------- top level ------------------------------------------------ */
  function analyse(gray, seg, S, wind, windDir) {
    var cls = seg.cls, probs = seg.probs;
    var oil = new Uint8Array(S * S), la = new Uint8Array(S * S);
    var counts = [0, 0, 0, 0, 0];
    for (var i = 0; i < S * S; i++) {
      counts[cls[i]]++;
      if (cls[i] === 1) oil[i] = 1;
      else if (cls[i] === 2) la[i] = 1;
    }
    var regions = groupRegions(oil, S);
    var dets = regions.map(function (r, j) {
      var m = measure(r, gray, cls, probs, S);
      var disc = discriminate(m, wind);
      m.id = "SLK-" + ("00" + (j + 1)).slice(-3);
      for (var k in disc) m[k] = disc[k];
      return m;
    });
    dets.sort(function (a, b) {
      return -(a.confidence * Math.sqrt(Math.min(a.area_km2, 25)))
             + (b.confidence * Math.sqrt(Math.min(b.area_km2, 25)));
    });

    var laRegions = groupRegions(la, S).map(function (r, j) {
      var m = measure(r, gray, cls, probs, S);
      return { id: "LAL-" + ("00" + (j + 1)).slice(-3), area_km2: m.area_km2,
               elongation: m.elongation, cy: m.cy, cx: m.cx, members: r };
    });

    return { class_counts: counts, detections: dets, lookalikes: laRegions,
             images: toDataURLs(gray, cls, S), size: S, wind_ms: wind };
  }

  return {
    analyse: analyse,
    backwardDrift: backwardDrift,
    windValidity: windValidity,
    PIXEL_M: PIXEL_M
  };
})();
