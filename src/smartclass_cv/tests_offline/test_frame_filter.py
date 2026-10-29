"""Uji offline FrameFilter (tanpa ROS):  python3 test_frame_filter.py"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from smartclass_cv.frame_filter import FrameFilter  # noqa: E402

rng = np.random.default_rng(0)
base = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


def noisy(img, sigma=2):
    n = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


# 1) tanpa filter: semua lolos
f = FrameFilter()
assert all(f.decide(base)[0] for _ in range(30))

# 2) decimate_n=3 dari 30 frame -> tepat 10 lolos, frame ke-1,4,7,... yang lolos
f = FrameFilter(decimate_n=3)
kept = [i for i in range(30) if f.decide(base)[0]]
assert kept == list(range(0, 30, 3)), kept

# 3) adegan statis (hanya noise kamera) -> hampir semua dibuang, kecuali yang pertama
f = FrameFilter(use_similarity=True, threshold=30.0, max_skip=0)
res = [f.decide(noisy(base))[0] for _ in range(100)]
assert res[0] and sum(res) == 1, sum(res)

# 4) heartbeat: adegan statis tetap menghasilkan update tiap max_skip+1 frame
f = FrameFilter(use_similarity=True, threshold=30.0, max_skip=10)
res = [f.decide(noisy(base))[0] for _ in range(111)]
assert sum(res) == 1 + 10, sum(res)       # pertama + 10 heartbeat

# 5) perubahan nyata (orang masuk) -> lolos
f = FrameFilter(use_similarity=True, threshold=30.0, max_skip=0)
f.decide(base)
changed = base.copy()
changed[100:400, 150:450] = 255          # blok besar berubah
ok, why = f.decide(changed)
assert ok and why == "lolos", (ok, why)

# 6) perubahan bertahap lambat tetap terdeteksi (bandingkan ke frame TERAKHIR YANG LOLOS)
f = FrameFilter(use_similarity=True, threshold=30.0, max_skip=0)
f.decide(base)
img = base.astype(np.float32)
lolos = 0
for _ in range(60):
    img = np.clip(img + 1.5, 0, 255)       # drift pelan per frame
    lolos += f.decide(img.astype(np.uint8))[0]
assert lolos >= 1, "drift pelan tidak pernah terdeteksi"

# 7) decimate + similarity digabung: frame yang dibuang decimate tidak menghitung similarity
f = FrameFilter(decimate_n=2, use_similarity=True, threshold=30.0, max_skip=0)
for _ in range(20):
    f.decide(noisy(base))
assert f.counts["decimate"] == 10 and f.counts["lolos"] == 1 and f.counts["mirip"] == 9, f.counts

# 8) biaya per panggilan (640x480)
f = FrameFilter(use_similarity=True)
frm = noisy(base)
t = time.perf_counter()
for _ in range(200):
    f.decide(frm)
ms = (time.perf_counter() - t) / 200 * 1000
print(f"biaya decide() similarity: {ms:.2f} ms/frame (PC ini)")
print(f.summary())
print("SEMUA UJI LOLOS")
