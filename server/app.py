"""Server inferensi model (FastAPI).

Jalankan dari folder server/ :
    EXPRESSION_MODEL=/path/best.pt ACTIVITY_MODEL=/path/best.onnx \
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoint:
    GET  /health
    POST /expression   body = bytes gambar (jpg/png) -> {"detections":[{label,conf}], ...}
    POST /activity     body = bytes gambar (jpg/png) -> {"detections":[{cls,conf,x1,y1,x2,y2}], ...}
"""
import hmac
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from ultralytics import YOLO

if __package__:
    from .face_gallery import load_gallery
else:
    from face_gallery import load_gallery

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "src", "smartclass_cv", "models")
MODEL_PATHS = {
    "expression": os.environ.get("EXPRESSION_MODEL", os.path.join(_BASE, "best.pt")),
    "activity": os.environ.get("ACTIVITY_MODEL", os.path.join(_BASE, "best.onnx")),
}
CONF = {"expression": 0.5, "activity": 0.4}   # sama dengan node lokal
API_TOKEN = os.environ.get("API_TOKEN", "")   # kosong = tanpa autentikasi (hanya untuk tes lokal)

MODELS = {}
LOCKS = {name: threading.Lock() for name in [*MODEL_PATHS, "recognition"]}


def _load(path):
    return YOLO(path, task="detect") if path.endswith(".onnx") else YOLO(path)


@asynccontextmanager
async def lifespan(app):
    dummy = np.zeros((100, 100, 3), np.uint8)
    for name, path in MODEL_PATHS.items():
        try:
            m = _load(path)
            m(dummy, conf=CONF[name], verbose=False)   # warm-up
            MODELS[name] = m
            print(f"[server] {name}: {path} siap")
        except Exception as e:  # server tetap hidup; endpoint terkait 503
            print(f"[server] GAGAL memuat {name} ({path}): {e}")

    # Encoding wajah (dlib + face_recognition) -- opsional, hanya kalau terpasang
    try:
        import face_recognition
        face_recognition.face_encodings(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))  # warm-up
        MODELS["recognition"] = face_recognition
        print("[server] recognition: face_recognition siap")
    except Exception as e:
        print(f"[server] recognition tidak tersedia: {e}")
    app.state.face_gallery = None
    gallery_path = os.environ.get("FACE_GALLERY_DB")
    if gallery_path:
        try:
            from smartclass_cv.recognition_match import compare_faces_np, pick_name
            app.state.face_gallery = load_gallery(gallery_path)
            app.state.face_match = (compare_faces_np, pick_name)
            print("[server] face gallery siap")
        except (ImportError, OSError, ValueError, sqlite3.Error) as e:
            print(f"[server] face gallery tidak tersedia: {e}")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    ready = {key: key in MODELS for key in [*MODEL_PATHS, "recognition"]}
    ready["face_analysis"] = (
        ready["expression"] and ready["recognition"]
        and getattr(app.state, "face_gallery", None) is not None
    )
    return {"ok": True, "models": ready}


def _decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "gambar tidak valid")
    return img


def _check_token(request: Request):
    if not API_TOKEN:
        return
    given = request.headers.get("authorization", "")
    if not hmac.compare_digest(given, f"Bearer {API_TOKEN}"):
        raise HTTPException(401, "token tidak valid")


async def _run(name, request: Request, fn):
    _check_token(request)
    model = MODELS.get(name)
    if model is None:
        raise HTTPException(503, f"model '{name}' belum siap")
    data = await request.body()
    t0 = time.perf_counter()
    img = _decode(data)
    decode_ms = (time.perf_counter() - t0) * 1000

    def work():
        with LOCKS[name]:               # satu inferensi per model dalam satu waktu
            t = time.perf_counter()
            out = fn(model, img)
            return out, (time.perf_counter() - t) * 1000

    out, infer_ms = await run_in_threadpool(work)
    return {"detections": out, "infer_ms": infer_ms, "decode_ms": decode_ms}


def _expression(model, img):
    results = model(img, conf=CONF["expression"], verbose=False)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            out.append({"label": model.names[int(box.cls[0])],
                        "conf": float(box.conf[0])})
    return out


def _activity(model, img):
    results = model(img, conf=CONF["activity"], verbose=False)[0]
    out = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        out.append({"cls": int(box.cls[0]), "conf": float(box.conf[0]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


@app.post("/expression")
async def expression(request: Request):
    return await _run("expression", request, _expression)


@app.post("/activity")
async def activity(request: Request):
    return await _run("activity", request, _activity)


def _face_encoding(fr, img):
    """Encoding 128-d per wajah pada crop (BGR -> RGB, sama dengan recognition_node lokal)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return [e.tolist() for e in fr.face_encodings(rgb)]


@app.post("/face_encoding")
async def face_encoding(request: Request):
    # "detections" berisi list encoding (tiap elemen = 128 angka), pencocokan dilakukan di klien
    return await _run("recognition", request, _face_encoding)


@app.post("/face_analysis")
async def face_analysis(request: Request):
    """Analyze a crop containing one face. Matching uses the server gallery."""
    _check_token(request)
    gallery = getattr(request.app.state, "face_gallery", None)
    expression_model = MODELS.get("expression")
    if gallery is None or expression_model is None:
        raise HTTPException(503, "face analysis belum siap")
    names, references = gallery
    compare, choose_name = request.app.state.face_match

    def analyze(recognition_model, image):
        encodings = _face_encoding(recognition_model, image)
        if len(encodings) != 1:
            status = "no_face" if not encodings else "multiple_faces"
            return {"status": status, "identity": None, "expression": None}

        label = choose_name(compare(references, encodings[0]), names)
        identity = None
        if label != "Unknown":
            name, separator, nim = label.partition("_")
            identity = {"label": label, "name": name, "nim": nim if separator else None}

        with LOCKS["expression"]:
            expressions = _expression(expression_model, image)
        expression = max(expressions, key=lambda item: item["conf"], default=None)
        return {
            "status": "matched" if identity else "unknown",
            "identity": identity,
            "expression": expression,
        }

    result = await _run("recognition", request, analyze)
    analysis = result.pop("detections")
    return {**analysis, **result}
