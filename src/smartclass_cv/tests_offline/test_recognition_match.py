"""Uji offline pencocokan wajah:  python3 test_recognition_match.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from smartclass_cv.recognition_match import compare_faces_np, pick_name  # noqa: E402

rng = np.random.default_rng(0)
known = [rng.normal(size=128) * 0.1 for _ in range(6)]
names = ["Budi_111", "Budi_111", "Sari_222", "Sari_222", "Sari_222", "Tono_333"]

# 1) sama dengan rumus face_recognition.compare_faces (norm per baris <= tolerance)
for _ in range(200):
    enc = known[rng.integers(0, 6)] + rng.normal(size=128) * rng.uniform(0, 0.08)
    expected = list(np.linalg.norm(np.array(known) - enc, axis=1) <= 0.45)
    assert compare_faces_np(known, enc, 0.45) == expected

# 2) pick_name: suara terbanyak menang; tidak cocok -> Unknown
assert pick_name([False] * 6, names) == "Unknown"
assert pick_name([True, True, False, True, False, False], names) == "Budi_111"
assert pick_name([False, False, True, True, True, True], names) == "Sari_222"
assert pick_name([True, False, False, False, False, True], names) == "Budi_111"  # seri -> yang pertama

# 3) database kosong -> tidak ada kecocokan, tidak error
assert compare_faces_np([], rng.normal(size=128)) == []
assert pick_name([], []) == "Unknown"

# 4) encoding lewat JSON (seperti dari server) identik dengan aslinya
import json
enc = rng.normal(size=128)
back = np.asarray(json.loads(json.dumps(enc.tolist())), dtype=np.float64)
assert np.array_equal(enc, back)

print("SEMUA UJI LOLOS")
