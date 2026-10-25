"""Helper HTTP untuk memanggil model yang berjalan di server.

Node tetap berjalan di Raspberry Pi / PC; hanya inferensi model yang
dikirim ke server lewat HTTP (POST gambar -> JSON).
"""
import time
from collections import deque

import cv2
import requests


class RemoteError(Exception):
    """Gagal memanggil server (timeout, koneksi, status non-200, JSON rusak)."""


class RemoteModel:
    def __init__(self, url, timeout=1.0, image_format="jpg", jpeg_quality=90):
        if image_format not in ("jpg", "png"):
            raise ValueError("image_format harus 'jpg' atau 'png'")
        self.url = url
        self.timeout = float(timeout)
        self.ext = "." + image_format
        self.params = (
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            if image_format == "jpg"
            else []
        )
        self.session = requests.Session()  # koneksi dipakai ulang

    def infer(self, bgr):
        """Kirim satu gambar BGR. Return dict dari server + rtt_ms & encode_ms."""
        t0 = time.monotonic_ns()
        ok, buf = cv2.imencode(self.ext, bgr, self.params)
        if not ok:
            raise RemoteError("encode gambar gagal")
        payload = buf.tobytes()
        t1 = time.monotonic_ns()
        try:
            r = self.session.post(
                self.url,
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self.timeout,
            )
            r.raise_for_status()
            out = r.json()
        except (requests.RequestException, ValueError) as e:
            raise RemoteError(str(e)) from e
        t2 = time.monotonic_ns()
        out["encode_ms"] = (t1 - t0) / 1e6
        out["rtt_ms"] = (t2 - t1) / 1e6      # kirim + server + terima
        out["payload_bytes"] = len(payload)
        return out


class RttStats:
    """Ringkasan sederhana: median/P95 RTT, infer server, jumlah gagal."""

    def __init__(self, logger, name, every=50):
        self.logger = logger
        self.name = name
        self.every = every
        self.rtt = deque(maxlen=200)
        self.infer = deque(maxlen=200)
        self.ok = 0
        self.fail = 0

    @staticmethod
    def _pct(values, p):
        s = sorted(values)
        if not s:
            return float("nan")
        return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]

    def add(self, res):
        self.ok += 1
        self.rtt.append(res["rtt_ms"])
        if "infer_ms" in res:
            self.infer.append(res["infer_ms"])
        self._maybe_log()

    def add_fail(self, err):
        self.fail += 1
        self.logger.warning(f"[{self.name}] remote gagal: {err}")
        self._maybe_log()

    def _maybe_log(self):
        n = self.ok + self.fail
        if n and n % self.every == 0:
            self.logger.info(
                f"[{self.name}] n={n} ok={self.ok} gagal={self.fail} "
                f"rtt median={self._pct(self.rtt, .5):.1f}ms "
                f"p95={self._pct(self.rtt, .95):.1f}ms "
                f"infer_server median={self._pct(self.infer, .5):.1f}ms"
            )
