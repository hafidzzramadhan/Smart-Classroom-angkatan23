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
