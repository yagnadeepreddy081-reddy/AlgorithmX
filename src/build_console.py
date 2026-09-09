"""Inject the pipeline output into the console template and write the artifact."""

import json
import os
import sys

TEMPLATE = "web/console.template.html"
DATA = "out/incidents.json"
OUT = "web/samudra_console.html"


def main():
    with open(TEMPLATE) as f:
        html = f.read()
    with open(DATA) as f:
        payload = json.load(f)

    # trim things the page never reads, to keep the artifact small
    for inc in payload["incidents"]:
        for d in inc["detections"]:
            dr = d.get("drift")
            if dr:
                for fr in dr["frames"]:
                    fr.pop("r50_km", None)
            for c in d.get("candidates", []):
                c.pop("corridor_hits", None)

    if os.path.exists("out/library.json"):
        lib = json.load(open("out/library.json"))
        payload["library"] = lib

    if os.path.exists("out/samples.json"):
        payload["samples"] = json.load(open("out/samples.json"))["tiles"]

    if os.path.exists("out/recovery.json"):
        rec = json.load(open("out/recovery.json"))
        rec.pop("trials", None)
        payload["recovery"] = rec

    blob = json.dumps(payload, separators=(",", ":"))
    blob = blob.replace("</script>", "<\\/script>")
    html = html.replace("__DATA__", blob)

    # in-browser inference: the network weights and the two engine scripts are
    # inlined so the page has no runtime dependencies of any kind
    with open("web/weights.json") as f:
        wblob = f.read().replace("</script>", "<\\/script>")
    html = html.replace("__WEIGHTS__", wblob)
    for token, path in [("__ENGINE__", "web/engine.js"), ("__ANALYSE__", "web/analyse.js")]:
        with open(path) as f:
            html = html.replace(token, f.read())
    os.makedirs("web", exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print("wrote %s  %.2f MB" % (OUT, os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
