"""Bandingkan hasil model lokal vs server untuk gambar yang sama.

Status per gambar:
  OK     label & jumlah deteksi sama, selisih conf < --tol, selisih box <= --box-tol px
  GESER  label & jumlah sama, tapi angka (conf/box) bergeser lebih dari toleransi
  LABEL  label atau jumlah deteksi berbeda  -> ini yang perlu diselidiki

Contoh:
  python compare_local_remote.py --kind activity --model ../src/smartclass_cv/models/best.onnx \
      --fmt jpg --images "test_activity_internet/*.jpg"
PNG = lossless (selisih harus ~0, pakai --tol 1e-6). JPG = lossy (selisih kecil wajar).
"""
import argparse
import glob
import os
import sys

import cv2
import requests
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--kind", choices=["expression", "activity"], required=True)
p.add_argument("--model", required=True)
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--fmt", choices=["png", "jpg"], default="png")
p.add_argument("--quality", type=int, default=90)
p.add_argument("--tol", type=float, default=0.01, help="toleransi selisih confidence")
p.add_argument("--box-tol", type=int, default=2, help="toleransi selisih box (piksel)")
p.add_argument("--images", nargs="+", required=True)
a = p.parse_args()

conf = 0.5 if a.kind == "expression" else 0.4
model = YOLO(a.model, task="detect") if a.model.endswith(".onnx") else YOLO(a.model)
params = [int(cv2.IMWRITE_JPEG_QUALITY), a.quality] if a.fmt == "jpg" else []
files = sorted(f for pat in a.images for f in glob.glob(pat))

counts = {"OK": 0, "GESER": 0, "LABEL": 0}
max_conf_all, max_box_all = 0.0, 0
for f in files:
    img = cv2.imread(f)
    local = []
    for box in model(img, conf=conf, verbose=False)[0].boxes:
        c = int(box.cls[0])
        key = model.names[c] if a.kind == "expression" else c
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        local.append((key, float(box.conf[0]), (x1, y1, x2, y2)))
    ok, buf = cv2.imencode("." + a.fmt, img, params)
    token = os.environ.get("SMARTCLASS_API_TOKEN", "")
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    d = requests.post(f"{a.url}/{a.kind}", data=buf.tobytes(), headers=hdr, timeout=10).json()["detections"]
    remote = []
    for x in d:
        if a.kind == "expression":
            remote.append((x["label"], x["conf"], None))
        else:
            remote.append((x["cls"], x["conf"], (x["x1"], x["y1"], x["x2"], x["y2"])))

    if [l[0] for l in local] != [r[0] for r in remote]:
        status, cd, bd = "LABEL", 0.0, 0
    else:
        cd = max([abs(l[1] - r[1]) for l, r in zip(local, remote)] or [0.0])
        bd = max([max(abs(u - v) for u, v in zip(l[2], r[2]))
                  for l, r in zip(local, remote) if l[2] and r[2]] or [0])
        status = "OK" if cd < a.tol and bd <= a.box_tol else "GESER"
    counts[status] += 1
    max_conf_all, max_box_all = max(max_conf_all, cd), max(max_box_all, bd)
    print(f"{status:5s} {f} local={[(l[0], round(l[1], 4)) for l in local]} "
          f"remote={[(r[0], round(r[1], 4)) for r in remote]} dconf={cd:.5f} dbox={bd}px")

print(f"\n{len(files)} gambar ({a.fmt}): OK={counts['OK']} GESER={counts['GESER']} "
      f"LABEL={counts['LABEL']} | selisih conf maks={max_conf_all:.5f}, box maks={max_box_all}px")
sys.exit(1 if counts["LABEL"] else 0)
