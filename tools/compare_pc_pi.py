"""Bandingkan waktu per callback (t_total_ms) PC vs Raspberry Pi per node.

PC: file CSV hasil probe yang sudah ada di experiments/ (i7-3770, kamera Hikvision).
Pi: CSV hasil probe yang sama yang dijalankan di Pi, diberi nama <node>_pi*.csv
    di satu folder (misal ~/hasil_pi/expression_pi.csv).

Pakai:
  python3 compare_pc_pi.py --pc-dir ~/smartclass_ws/experiments --pi-dir ~/hasil_pi
Hasil dicetak sebagai tabel dan disimpan ke perbandingan_pc_pi.md di --pi-dir.
Hanya butuh Python 3 standar (tanpa pandas).
"""
import argparse
import csv
import glob
import os
import statistics

PC_FILES = {
    "gesture":     "gesture_pc_hikvision_14386616741255.csv",
    "recognition": "recognition_pc_hikvision_13253283618281.csv",
    "detection":   "detection_pc_hikvision_run02.csv",
    "activity":    "activity_pc_hikvision_2408765625041.csv",
    "face":        "face_pc_hikvision_13714347236728.csv",
    "expression":  "expression_pc_hikvision_11467902022263.csv",
}


def load(path, skip):
    vals = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i < skip:
                continue                       # buang panggilan awal (warm-up)
            if row.get("status", "ok") not in ("ok", ""):
                continue
            try:
                vals.append(float(row["t_total_ms"]))
            except (KeyError, ValueError):
                continue
    return vals


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]


def latest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc-dir", required=True)
    ap.add_argument("--pi-dir", required=True)
    ap.add_argument("--skip", type=int, default=1, help="baris awal yang dibuang")
    a = ap.parse_args()
    pc_dir, pi_dir = os.path.expanduser(a.pc_dir), os.path.expanduser(a.pi_dir)

    rows = []
    for node, pc_name in PC_FILES.items():
        pc = load(os.path.join(pc_dir, pc_name), a.skip)
        pi_file = latest(os.path.join(pi_dir, f"{node}_pi*.csv"))
        pi = load(pi_file, a.skip) if pi_file else []
        rows.append((node, pc, pi))

    head = ("| Node | n PC | n Pi | Median PC (ms) | Median Pi (ms) | Rasio Pi/PC "
            "| P95 PC (ms) | P95 Pi (ms) |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [head, sep]
    for node, pc, pi in rows:
        mpc = statistics.median(pc) if pc else None
        mpi = statistics.median(pi) if pi else None
        ratio = f"{mpi / mpc:.1f}x" if mpc and mpi else "-"
        f = lambda v: f"{v:.0f}" if v is not None else "belum"
        lines.append(
            f"| {node} | {len(pc)} | {len(pi)} | {f(mpc)} | {f(mpi)} | {ratio} | "
            f"{f(pct(pc, .95) if pc else None)} | {f(pct(pi, .95) if pi else None)} |")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(pi_dir, "perbandingan_pc_pi.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\nTersimpan: {os.path.join(pi_dir, 'perbandingan_pc_pi.md')}")


if __name__ == "__main__":
    main()
