import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

from enroll_face import encode_photo, enroll
from face_gallery import load_gallery


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "faces.sqlite3"
        with sqlite3.connect(self.db) as db:
            db.execute("CREATE TABLE face_encodings (id INTEGER PRIMARY KEY, label TEXT NOT NULL, encoding TEXT NOT NULL)")
            db.execute("INSERT INTO face_encodings(label, encoding) VALUES (?, ?)",
                       ("Existing_001", json.dumps([0.0] * 128)))

    def test_append_duplicate_and_api_loader(self):
        self.assertTrue(enroll(self.db, "Hafidz Ramadhan", "002", [0.1] * 128)[0])
        self.assertFalse(enroll(self.db, "Hafidz Ramadhan", "002", [0.1] * 128)[0])
        names, vectors = load_gallery(self.db)
        self.assertEqual(names, ("Existing_001", "Hafidz Ramadhan_002"))
        self.assertEqual(vectors.shape, (2, 128))
        self.assertTrue((vectors[0] == 0).all())

    def test_invalid_reference_leaves_database_unchanged(self):
        for vector in ([1] * 127, [float("nan")] * 128):
            with self.assertRaises(ValueError):
                enroll(self.db, "Hafidz", "002", vector)
        with self.assertRaises(ValueError):
            enroll(self.db, "Hafidz_Test", "002", [1] * 128)
        self.assertEqual(len(load_gallery(self.db)[0]), 1)

    def test_missing_database_not_created(self):
        missing = self.db.with_name("wrong.sqlite3")
        with self.assertRaises(sqlite3.OperationalError):
            enroll(missing, "Hafidz", "002", [1] * 128)
        self.assertFalse(missing.exists())

    def test_zero_or_multiple_faces_rejected(self):
        for locations in ([], [(0, 1, 1, 0), (2, 3, 3, 2)]):
            backend = Mock()
            backend.face_locations.return_value = locations
            with self.assertRaises(ValueError):
                encode_photo("photo.jpg", backend)
            backend.face_encodings.assert_not_called()

    def test_single_face_encoded(self):
        backend = Mock()
        backend.face_locations.return_value = [(0, 1, 1, 0)]
        backend.face_encodings.return_value = [[0.1] * 128]
        self.assertEqual(len(encode_photo("photo.jpg", backend)), 128)


if __name__ == "__main__":
    unittest.main()
