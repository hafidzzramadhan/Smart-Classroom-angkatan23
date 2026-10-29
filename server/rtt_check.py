"""Ukur RTT ke server inferensi dengan satu gambar.

  export SMARTCLASS_API_TOKEN=...        # kalau server memakai token
  python3 rtt_check.py http://IP:8100/expression wajah.png [jumlah]
"""
import os
import statistics
import sys
import time

import requests

url, path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
data = open(path, "rb").read()
token = os.environ.get("SMARTCLASS_API_TOKEN", "")
s = requests.Session()
if token:
    s.headers["Authorization"] = f"Bearer {token}"

rtt, infer, gagal = [], [], 0
for i in range(n + 1):
    t = time.perf_counter()
    try:
        r = s.post(url, data=data, timeout=10)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        gagal += 1
        print("gagal:", e)
        continue
    if i:                                    # buang panggilan pertama
        rtt.append((time.perf_counter() - t) * 1000)
        infer.append(j["infer_ms"])

if rtt:
    rtt.sort()
    print(f"payload {len(data)} B | n={len(rtt)} gagal={gagal} | "
          f"rtt median {statistics.median(rtt):.1f} ms, p95 {rtt[int(.95*(len(rtt)-1))]:.1f} ms | "
          f"infer server median {statistics.median(infer):.1f} ms")
