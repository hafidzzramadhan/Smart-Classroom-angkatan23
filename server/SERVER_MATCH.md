# Uji pencocokan wajah di server

Mode **Server match (nama + ekspresi)** mengirim satu crop wajah ke
`POST /face_analysis`. Server mencari identitas di database SQLite dan
mengembalikan identitas, ekspresi, dan durasi proses. Merge menerima satu
anotasi gabungan melalui `/cv/bondingbox/face_analysis`.

## Siapkan server uji

Jalankan dari root repo. Siapkan `/path/enrollment/encodings.pickle` dari
pendaftaran wajah yang dipercaya. Siapkan direktori model berisi `best.pt`
dan, jika Activity digunakan, `best.onnx`.

```bash
docker build --build-arg WITH_RECOGNITION=1 -f Dockerfile.server -t smartclass-infer:server-match .
gallery_dir="$HOME/smartclass-face-gallery"
mkdir -p "$gallery_dir"

docker run --rm \
  --mount "type=bind,source=/path/enrollment/encodings.pickle,target=/input/encodings.pickle,readonly" \
  --mount "type=bind,source=$gallery_dir,target=/gallery" \
  smartclass-infer:server-match \
  python face_gallery.py /input/encodings.pickle /gallery/faces.sqlite3
```

Importer membuat database baru. Satu orang bisa mempunyai beberapa vektor.
Format label mengikuti data yang sudah ada, misalnya `Nama_NIM`. Simpan
database dan file enrollment di luar Git. Server membaca database saat
startup. Restart layanan setelah referensi diperbarui.

Contoh menjalankan container pada port uji yang bisa dihubungkan ke reverse
proxy. Jika proxy menyediakan autentikasi API, samakan `API_TOKEN` di server
dan `SMARTCLASS_API_TOKEN` di Pi.

```bash
docker run -d --name smartclass-infer-match \
  -p 127.0.0.1:8101:8000 \
  --mount "type=bind,source=/path/models,target=/models,readonly" \
  --mount "type=bind,source=$gallery_dir,target=/gallery,readonly" \
  -e FACE_GALLERY_DB=/gallery/faces.sqlite3 \
  -e API_TOKEN \
  smartclass-infer:server-match

curl -fsS http://127.0.0.1:8101/health
```

Pastikan `models.face_analysis` bernilai `true`. Kirim satu crop dari wajah
uji yang sudah didaftarkan. Bila token disetel, sertakan header Authorization.

```bash
curl -fsS -H "Authorization: Bearer ${API_TOKEN:-}" \
  --data-binary @wajah_uji.jpg http://127.0.0.1:8101/face_analysis
```

Contoh hasil jika cocok dan ekspresi terdeteksi:

```json
{
  "status": "matched",
  "identity": {"label": "Hafidz_123", "name": "Hafidz", "nim": "123"},
  "expression": {"label": "sad", "conf": 0.86},
  "infer_ms": 570.2,
  "decode_ms": 1.4
}
```

Wajah yang tidak cocok menghasilkan `unknown` dengan `identity: null`.
Untuk crop kosong atau crop yang berisi beberapa wajah, API mengembalikan
`no_face` atau `multiple_faces`. Model dapat mengembalikan
`expression: null` bila tidak ada hasil ekspresi yang melewati ambang.

## Jalankan dari Pi

```bash
cd ~/smartclass_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select smartclass_cv
source install/setup.bash
source .venv/bin/activate
python smartclass_web_launcher.py --source 0 --decimate-n 3 --server-url 'https://URL-UJI-DARI-MAS-FIKRI'
```

Di launcher, pilih **Server match (nama + ekspresi)** dan klik **Start semua**.
Node Recognition menjalankan satu request per crop dan mengirim anotasi
gabungan ke Merge. Contoh pengecekan hasil dari terminal ROS:

```bash
timeout 30s ros2 topic echo /cv/bondingbox/face_analysis --field label --once
```

Database ini adalah daftar referensi untuk uji awal. Pemilihan peserta
berdasarkan ruang dan waktu, sinkronisasi SIS, dan pencatatan absensi pada
jalur baru perlu integrasi terpisah.

## Tambah wajah baru dari foto ke SQLite yang sudah ada

Gunakan `server/enroll_face.py`. Skrip ini memerlukan satu wajah per foto,
menambahkan label `Nama_NIM` dan encoding 128 angka tanpa menghapus data lama.
Nama dan NIM tidak boleh memakai underscore karena pemisah identitas API
menggunakan underscore. Foto/encoding harus dibuat dengan model encoding
wajah yang sama dengan API. Foto tidak disalin ke database; hanya encoding.
Skrip menolak database yang tidak ada dan tidak menggandakan label+encoding
identik. Foto berbeda untuk orang yang sama dapat menjadi referensi tambahan.

### Server Mas Fikri: tanpa rebuild Docker

Jalankan dari root repo yang sudah berisi `server/enroll_face.py`. Container
layanan yang sudah ada bernama `smartclass-infer-match`. Untuk instalasi ini,
SQLite berada di `/root/smartclass-face-gallery/faces.sqlite3` pada host dan
`/gallery/faces.sqlite3` di container. Sesuaikan bila deployment berbeda.

1. Isi lokasi foto dan NIM asli melalui prompt (jangan mengetik contoh NIM):

```bash
read -r -p 'Lokasi lengkap foto di server: ' enrollment_photo
read -r -p 'Nama lengkap: ' enrollment_name
read -r -p 'NIM: ' enrollment_nim
```

2. Jalankan dari terminal yang sama. Container sementara memakai image persis
   dari layanan aktif, sehingga memakai instalasi dlib yang sama. Mount foto
   dan skrip read-only; mount galeri writable untuk menambahkan referensi.
   `--mount` akan gagal jika file sumber tidak ada.

```bash
enrollment_image=$(docker inspect --format '{{.Image}}' smartclass-infer-match)
docker run --rm --network none \
  --mount "type=bind,source=$PWD/server/enroll_face.py,target=/app/enroll_face.py,readonly" \
  --mount "type=bind,source=$enrollment_photo,target=/input/photo,readonly" \
  --mount 'type=bind,source=/root/smartclass-face-gallery,target=/gallery' \
  "$enrollment_image" \
  python /app/enroll_face.py \
  --image /input/photo \
  --database /gallery/faces.sqlite3 \
  --name "$enrollment_name" --nim "$enrollment_nim"
```

Berhasil bila muncul `DITAMBAHKAN` atau `SUDAH ADA`, panjang encoding 128,
dan jumlah referensi. Jika gagal, berhenti dan periksa error; jangan lanjut
restart. Tidak ada encoding mentah yang dicetak ke log.

3. Setelah pendaftaran berhasil, restart layanan untuk memuat ulang galeri:

```bash
docker restart smartclass-infer-match
docker logs --tail 30 smartclass-infer-match
curl -fsS http://127.0.0.1:8101/health
```

Tunggu startup selesai. Pastikan `models.face_analysis` adalah `true`.
Alamat lokal ini untuk pengecekan di server; Pi tetap memakai alamat API
publik dari pengelola (sebelumnya `http://103.28.161.71:8085`).

4. Di launcher Pi, stop Expression dan Recognition lama, pilih **Server match
   (nama + ekspresi)**, isi URL server yang benar, lalu start Recognition.
   Frame, Face, Merge dan viewer tetap diperlukan. Expression terpisah tidak
   perlu dinyalakan di mode ini. Hadapkan wajah terdaftar ke kamera dan cek:

```bash
ros2 topic echo /cv/bondingbox/face_analysis --field label --once
```

Uji dengan kamera/foto berbeda dari foto pendaftaran, lalu dengan wajah yang
belum terdaftar. Keberhasilan pendaftaran tidak menjamin setiap foto akan
cocok; pencahayaan, pose, kualitas crop dan ambang pencocokan memengaruhi hasil.
Ini tes SQLite; integrasi MySQL/MariaDB adalah pekerjaan berikutnya.

### Validasi skrip tanpa model

```bash
python -m unittest discover -s server -p 'test_enroll_face.py' -v
```

Tes ini memeriksa penambahan, pencegahan duplikat, validasi vektor, penolakan
foto nol/banyak wajah dengan backend tiruan, serta kompatibilitas dengan
pembaca galeri API. Uji dlib/foto asli dan Docker tetap dijalankan di server.
