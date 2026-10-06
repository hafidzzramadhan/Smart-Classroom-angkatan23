import sqlite3
# Logic database sesuai file asli
conn = sqlite3.connect("absensi.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS absensi (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nim TEXT,
    nama TEXT,
    tanggal TEXT,
    waktu TEXT
)
""")
conn.commit()
conn.close()
print("Database absensi.db berhasil dibuat!")

