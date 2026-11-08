"""Register one reference photo in the SQLite gallery used by /face_analysis."""

import argparse
import json
from pathlib import Path
import sqlite3

from face_gallery import validate_reference


def encode_photo(path, face_backend=None):
    if face_backend is None:
        import face_recognition as face_backend
    image = face_backend.load_image_file(str(path))
    locations = face_backend.face_locations(image, model="hog")
    if len(locations) != 1:
        raise ValueError(f"Foto harus berisi tepat 1 wajah; terdeteksi {len(locations)}")
    vectors = face_backend.face_encodings(image, known_face_locations=locations)
    if len(vectors) != 1:
        raise ValueError("Encoding wajah gagal dibuat")
    return vectors[0]


def enroll(database, name, nim, encoding):
    name, nim = name.strip(), nim.strip()
    if not name or not nim or "_" in name or "_" in nim:
        raise ValueError("Nama dan NIM wajib diisi; nama/NIM tidak boleh mengandung underscore")
    label, vector = validate_reference(f"{name}_{nim}", encoding)
    encoded = json.dumps(vector.tolist())
    # mode=rw deliberately refuses a typo or missing database; never creates another DB.
    uri = Path(database).resolve().as_uri() + "?mode=rw"
    with sqlite3.connect(uri, uri=True, timeout=30) as db:
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("SELECT label, encoding FROM face_encodings").fetchall()
        for old_label, old_encoding in rows:
            if old_label == label and json.loads(old_encoding) == vector.tolist():
                return False, label, len(rows)
        db.execute("INSERT INTO face_encodings (label, encoding) VALUES (?, ?)",
                   (label, encoded))
        return True, label, len(rows) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--nim", required=True)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Foto tidak ditemukan: {args.image}")
    if not args.database.is_file():
        parser.error(f"Database yang sudah ada tidak ditemukan: {args.database}")
    try:
        added, label, count = enroll(args.database, args.name, args.nim,
                                     encode_photo(args.image))
    except (ValueError, OSError, sqlite3.Error, ImportError) as exc:
        parser.exit(1, f"GAGAL: {exc}\n")
    print(f"{'DITAMBAHKAN' if added else 'SUDAH ADA'}: {label}")
    print(f"Panjang encoding: 128; total referensi: {count}")
    print("Restart container API agar referensi dimuat ulang.")


if __name__ == "__main__":
    main()
