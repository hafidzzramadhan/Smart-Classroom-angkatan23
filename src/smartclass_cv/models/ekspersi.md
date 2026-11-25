# 🎓 Smart Classroom: Sistem Deteksi Ekspresi Wajah Berbasis YOLO & ONNX (Raspberry Pi 5)

Dokumentasi resmi pengembangan dan komparasi model *Computer Vision* untuk deteksi ekspresi wajah siswa secara *real-time* di lingkungan *Smart Classroom*. Proyek ini membandingkan model dasar (*baseline*) dengan dua varian model kustom berbasis **YOLO11n** dan **YOLO26n** yang dioptimalkan menggunakan **ONNX Runtime** untuk eksekusi ringan di perangkat *Edge Computing* (Raspberry Pi 5).

---

## 📷 Gambaran Umum Proyek

Sistem ini dirancang untuk membantu pengajar memantau kondisi emosional dan tingkat keterlibatan (*engagement*) siswa di kelas secara non-invasif. Fokus utama pengujian ini adalah mentransformasi model deteksi ekspresi agar dapat berjalan **mulus (25–30 Display FPS)** di mini komputer **Raspberry Pi 5** tanpa mengorbankan akurasi klasifikasi.

---

## 🏷️ Pemetaan & Penyederhanaan Kelas

Model Lama (*Baseline*) menggunakan skema 8 kelas berbahasa Inggris. Model kustom disederhanakan menjadi **5 kelas emosi utama** berbahasa Indonesia untuk menghilangkan ambiguitas dan kelas yang tidak relevan di lingkungan sekolah.

| Indeks Baru | Kelas Kustom (Indonesia) | Indeks Lama | Pasangan Kelas Lama (Inggris) | Status Pemetaan |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Marah** | 0 | `anger` | Digunakan |
| **1** | **Netral** | 5 | `neutral` | Digunakan (Fokus Utama) |
| **2** | **Sedih** | 6 | `sad` | Digunakan |
| **3** | **Senang** | 4 | `happy` | Digunakan |
| **4** | **Takut** | 3 | `fear` | Digunakan |
| - | *(Dieliminasi)* | 1 | `content` | Dihapus (Redundan) |
| - | *(Dieliminasi)* | 2 | `disgust` | Dihapus (Sering Salah Tebak) |
| - | *(Dieliminasi)* | 7 | `surprise` | Dihapus (Redundan) |

---

## 📊 Hasil Komparasi & Benchmark 3 Model ONNX

Pengujian dilakukan pada CPU murni (`CPUExecutionProvider` ONNX Runtime) menggunakan resolusi input `640x640` piksel sebanyak 30 iterasi per model.

| Parameter Komparasi | Model Lama (`expression.onnx`) | Model v1 (`best-hakim.onnx`) | Model v2 (`best.onnx`) | Keunggulan Model Terpilih (v2) |
| :--- | :---: | :---: | :---: | :--- |
| **Arsitektur Model** | YOLO Eksternal | YOLO11 Nano | **YOLO26 Nano** | Modern (NMS-Free & 5,4 GFLOPs) |
| **Ukuran File (`.onnx`)** | 36,18 MB | 10,17 MB | **9,40 MB** | **74,0% Lebih Ringan** (Hemat RAM) |
| **Jumlah Kelas** | 8 Kelas (Inggris) | 5 Kelas (Indonesia) | **5 Kelas (Indonesia)** | Lebih Fokus & Bebas Bias |
| **mAP50 (Akurasi Standar)** | Tidak Diketahui | 94,10% | **96,60%** | **Meningkat +2,50%** |
| **mAP50-95 (Presisi Kotak)** | Tidak Diketahui | 75,50% | **76,50%** | **Meningkat +1,00%** |
| **Box Precision (Ketepatan)** | Tidak Diketahui | 85,80% | **92,50%** | **Meningkat +6,70%** |
| **Recall (Sensitivitas)** | Tidak Diketahui | 88,00% | **90,10%** | **Meningkat +2,10%** |
| **Waktu Inferensi (CPU)** | 498,67 ms | 158,17 ms | **138,28 ms** | **3,6x Lebih Cepat** (Hemat 360 ms) |
| **FPS Murni (CPU Murni)** | 2,0 FPS | 6,3 FPS | **7,2 FPS** | **3,6x Lebih Responsif** |
| **Display FPS di Pi 5** | Patah-patah (<5 FPS) | Cukup (15–20 FPS) | **Mulus (25–30 FPS)** | Sangat Layak (Frame Skipping) |

---

## 🎯 Rincian Metrik Evaluasi Model Terbaik (YOLO26n)

Hasil validasi terperinci per kelas pada 230 sampel gambar uji (Best Epoch 79):

| Kelas Ekspresi | Jumlah Gambar | Jumlah Sampel | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keseluruhan (All)** | **230** | **230** | **92,5%** | **90,1%** | **96,6%** | **76,5%** |
| **Marah** | 46 | 46 | 83,9% | 91,3% | 95,9% | 79,0% |
| **Netral** | 46 | 46 | 86,5% | 87,0% | 92,7% | 72,3% |
| **Sedih** | 46 | 46 | 95,4% | 89,1% | 96,8% | 73,5% |
| **Senang** | 46 | 46 | 98,9% | 97,8% | 99,3% | 79,3% |
| **Takut** | 46 | 46 | 97,5% | 85,3% | 98,3% | 78,2% |

---

## 📈 Analisis Perbaikan Per Kelas (YOLO11n vs YOLO26n)

| Kelas Ekspresi | Precision (YOLO11n) | Precision (YOLO26n) | Selisih Precision | mAP50 (YOLO11n) | mAP50 (YOLO26n) | Selisih mAP50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Marah** | 85,3% | **83,9%** | -1,4% | 92,7% | **95,9%** | +3,2% |
| **Netral** | 77,6% | **86,5%** | **+8,9%** | 90,0% | **92,7%** | +2,7% |
| **Sedih** | 87,3% | **95,4%** | **+8,1%** | 95,4% | **96,8%** | +1,4% |
| **Senang** | 88,7% | **98,9%** | **+10,2%** | 98,3% | **99,3%** | +1,0% |
| **Takut** | 89,9% | **97,5%** | **+7,6%** | 94,4% | **98,3%** | +3,9% |
| **Rata-rata (All)** | 85,8% | **92,5%** | **+6,7%** | 94,1% | **96,6%** | **+2,5%** |

* **Penanganan Bias Prediksi:** Lonjakan Precision pada kelas **Netral (+8,9%)** dan **Sedih (+8,1%)** secara drastis menghilangkan kesalahan prediksi (*false positive*) saat kamera menangkap wajah ber-ekspresi datar.

---

## 📁 Struktur Direktori Proyek

* `expression.onnx` — Model Lama Baseline (36,18 MB, 8 Kelas)
* `best-hakim.onnx` — Model Baru v1 YOLO11n (10,17 MB, 5 Kelas)
* `best.onnx` — Model Baru v2 YOLO26n (9,40 MB, 5 Kelas) — **Model Utama Dipakai**
* `README.md` — Dokumen Laporan Pengujian & Komparasi Sistem

---

## ⚡ Strategi Deployment di Raspberry Pi 5

Untuk mencapai tampilan kamera yang mulus di Raspberry Pi 5 tanpa rasa tersendat/patah-patah:

1. **Frame Skipping:** Jalankan proses inferensi AI setiap 3 *frame* sekali, sementara *frame* jeda tetap menampilkan hasil deteksi terakhir (*Box Persistence*). Teknik ini membuat tampilan layar tetap berjalan pada 25–30 Display FPS.
2. **Resolusi Input Kamera:** Atur resolusi tangkapan kamera ke `640x480` agar beban pemrosesan USB/CSI kamera di Raspberry Pi 5 tetap ringan.
3. **Pendinginan Aktif:** Pastikan Raspberry Pi 5 dilengkapi dengan *Active Cooler* (kipas + *heatsink*) untuk mencegah *thermal throttling* saat pengujian berdurasi panjang di ruang kelas.

---

## 📌 Kesimpulan Pemilihan Model

Model **`best.onnx` (YOLO26n)** resmi dipilih sebagai model final untuk sistem *Smart Classroom* berbasis Raspberry Pi 5 karena:

1. **Akurasi mAP50 Tertinggi:** **96,60%** (+2,5% dibanding YOLO11n).
2. **Presisi Tangguh (Box Precision):** **92,50%** (Mengatasi kekeliruan prediksi pada ekspresi Netral & Sedih).
3. **Ukuran File Paling Ringan:** **9,40 MB** (74% lebih kecil dibanding model lama).
4. **Waktu Eksekusi Tercepat:** **138,28 ms** (3,6x lebih cepat dari model lama).
5. **Performa Edge Computing:** Sangat layak dan mulus dijalankan pada Raspberry Pi 5.