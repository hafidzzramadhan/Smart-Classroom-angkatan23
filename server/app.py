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
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from ultralytics import YOLO

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "src", "smartclass_cv", "models")
MODEL_PATHS = {
    "expression": os.environ.get("EXPRESSION_MODEL", os.path.join(_BASE, "best.pt")),
    "activity": os.environ.get("ACTIVITY_MODEL", os.path.join(_BASE, "best.onnx")),
}
CONF = {"expression": 0.5, "activity": 0.4}   # sama dengan node lokal
API_TOKEN = os.environ.get("API_TOKEN", "")   # kosong = tanpa autentikasi (hanya untuk tes lokal)

MODELS = {}
LOCKS = {name: threading.Lock() for name in MODEL_PATHS}


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
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "models": {k: (k in MODELS) for k in MODEL_PATHS}}


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
