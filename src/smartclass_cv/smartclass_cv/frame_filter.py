"""Filter frame di sisi sumber (tanpa ROS, jadi bisa diuji offline).

Dipakai oleh frame_node untuk menurunkan laju frame yang masuk ke pipeline
supaya mendekati kapasitas pemrosesan (solusi 1 untuk delay).

Dua mekanisme, bisa dipakai sendiri-sendiri atau digabung:
  decimate_n      : teruskan 1 dari tiap N frame (deterministik, mudah diulang).
  use_similarity  : buang frame yang hampir sama dengan frame TERAKHIR YANG
                    DITERUSKAN (bukan frame sebelumnya), dihitung pada gambar
                    kecil agar murah. `max_skip` memaksa satu frame lolos setelah
                    sekian frame berturut-turut dibuang (heartbeat), supaya
                    adegan statis tetap menghasilkan update berkala.

Filter di sumber menjaga semua node melihat frame yang sama, jadi Merge masih
bisa menggabungkan hasil per frame.
"""
import cv2
import numpy as np


class FrameFilter:
    def __init__(self, decimate_n=1, use_similarity=False, threshold=30.0,
                 max_skip=15, small_size=(80, 60)):
        self.decimate_n = max(1, int(decimate_n))
        self.use_similarity = bool(use_similarity)
        self.threshold = float(threshold)
        self.max_skip = max(0, int(max_skip))   # 0 = tanpa heartbeat
        self.small_size = tuple(small_size)

        self._tick = 0
        self._ref = None            # gambar kecil dari frame terakhir yang diteruskan
        self._skipped = 0           # frame mirip berturut-turut yang sudah dibuang
        self.last_mse = None
        self.counts = {"lolos": 0, "decimate": 0, "mirip": 0, "heartbeat": 0}

    def _small_gray(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, self.small_size, interpolation=cv2.INTER_AREA)
        return g.astype(np.float32)

    def decide(self, frame):
        """Return (teruskan: bool, alasan: str). Dipanggil sekali per frame terbaca."""
        self._tick += 1

        # 1) decimation (jangan hitung similarity untuk frame yang sudah pasti dibuang)
        if self.decimate_n > 1 and (self._tick - 1) % self.decimate_n != 0:
            self.counts["decimate"] += 1
            return False, "decimate"

        # 2) similarity terhadap frame terakhir yang diteruskan
        if self.use_similarity:
            small = self._small_gray(frame)
            if self._ref is None:
                self._ref, self._skipped = small, 0
                self.counts["lolos"] += 1
                return True, "lolos"

            self.last_mse = float(np.mean((small - self._ref) ** 2))
            if self.last_mse > self.threshold:
                self._ref, self._skipped = small, 0
                self.counts["lolos"] += 1
                return True, "lolos"

            # heartbeat: setelah max_skip frame mirip berturut-turut dibuang,
            # frame berikutnya dipaksa lolos.
            if self.max_skip and self._skipped >= self.max_skip:
                self._ref, self._skipped = small, 0
                self.counts["heartbeat"] += 1
                return True, "heartbeat"

            self._skipped += 1
            self.counts["mirip"] += 1
            return False, "mirip"

        self.counts["lolos"] += 1
        return True, "lolos"

    def summary(self):
        c = self.counts
        total = sum(c.values())
        kept = c["lolos"] + c["heartbeat"]
        pct = 100.0 * kept / total if total else 0.0
        mse = "-" if self.last_mse is None else f"{self.last_mse:.1f}"
        return (f"[filter] total={total} diteruskan={kept} ({pct:.0f}%) "
                f"lolos={c['lolos']} heartbeat={c['heartbeat']} "
                f"dibuang: decimate={c['decimate']} mirip={c['mirip']} | mse_terakhir={mse}")
