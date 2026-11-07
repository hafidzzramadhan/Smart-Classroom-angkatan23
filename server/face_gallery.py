"""Load face references from SQLite or import an existing trusted enrollment file."""

import argparse
import json
from pathlib import Path
import pickle
import sqlite3

import numpy as np


def validate_reference(label, encoding):
    vector = np.asarray(encoding, dtype=np.float64)
    if not isinstance(label, str) or not label.strip() or label == "Unknown":
        raise ValueError("Invalid face reference label")
    if vector.shape != (128,) or not np.isfinite(vector).all():
        raise ValueError("Face reference must contain 128 finite numbers")
    return label, vector


def load_gallery(path):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        references = [
            validate_reference(label, json.loads(encoding))
            for label, encoding in db.execute(
                "SELECT label, encoding FROM face_encodings ORDER BY id"
            )
        ]
    if not references:
        raise ValueError("No face references registered")
    names, encodings = zip(*references)
    return names, np.stack(encodings)


def import_gallery(pickle_path, database_path):
    with pickle_path.open("rb") as source:
        data = pickle.load(source)
    rows = [
        (label, json.dumps(vector.tolist()))
        for label, vector in (
            validate_reference(label, encoding)
            for label, encoding in zip(data["names"], data["encodings"], strict=True)
        )
    ]
    if not rows:
        raise ValueError("No face references to import")
    # Exclusive creation prevents replacing an existing enrollment database.
    with database_path.open("xb"):
        pass
    try:
        with sqlite3.connect(database_path) as db:
            db.execute(
                "CREATE TABLE face_encodings ("
                "id INTEGER PRIMARY KEY, label TEXT NOT NULL, encoding TEXT NOT NULL)"
            )
            db.executemany(
                "INSERT INTO face_encodings (label, encoding) VALUES (?, ?)", rows
            )
    except Exception:
        database_path.unlink()
        raise
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pickle", type=Path, help="Trusted encodings.pickle created by your team")
    parser.add_argument("database", type=Path, help="Path for a new SQLite database")
    args = parser.parse_args()
    print(f"Imported {import_gallery(args.pickle, args.database)} face references")
