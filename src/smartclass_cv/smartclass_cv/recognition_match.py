"""Pencocokan wajah tanpa dlib dan tanpa ROS (bisa diuji offline).

Rumus sama dengan face_recognition.compare_faces:
    jarak = norm(known - encoding) per baris, cocok kalau jarak <= tolerance.
"""
import numpy as np


def compare_faces_np(known_encodings, encoding, tolerance=0.45):
    known = np.asarray(known_encodings)
    if known.shape[0] == 0:
        return []
    dist = np.linalg.norm(known - np.asarray(encoding), axis=1)
    return list(dist <= tolerance)


def pick_name(matches, names):
    """Logika asli recognition_node: nama dengan jumlah cocok terbanyak."""
    name = "Unknown"
    if True in matches:
        matched_idxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matched_idxs:
            n = names[i]
            counts[n] = counts.get(n, 0) + 1
        name = max(counts, key=counts.get)
    return name
