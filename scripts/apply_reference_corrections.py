"""Apply source-verified corrections to the public-beta testset references.

The originals under tests/fixtures/ are untouched. Each correction below was
checked against the raw OCR text in .agent/assets/kb:

- d1-03, d2-03: the perwalian table on Wadek p.68 says IPS<=3.0 -> 20 SKS and
  IPS>3.0 -> 24 SKS. The original reference said ">= 3.0 -> 24", which is wrong
  at the boundary.
- d2-08: the yudisium table on Wadek p.84 maps 3.01-3.50 to "Sangat Memuaskan",
  2.75-3.00 to "Memuaskan", and 3.51-4.00 to "Dengan Pujian (Cum Laude)".
  The original reference shifted every label by one row.
- d2-05, d2-15: Wadek p.93 requires "Sks lulus minimal 70" (SKS), not 70 percent.
"""

from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("tests/fixtures/eval/public_beta_v1.jsonl")
TARGET_V1 = Path("tests/fixtures/eval/public_beta_v1_1.jsonl")
TARGET_V2 = Path("tests/fixtures/eval/public_beta_v1_2.jsonl")

CORRECTIONS = {
    "d1-03": (
        "Berdasarkan aturan perwalian reguler, mahasiswa dengan Indeks Prestasi Semester "
        "(IPS) lebih dari 3.0 diperbolehkan mengambil beban studi maksimal 24 SKS, "
        "sedangkan mahasiswa dengan IPS 3.0 atau kurang hanya dapat mengambil maksimal 20 SKS."
    ),
    "d2-03": (
        "Batas pengambilan SKS per semester ditentukan oleh IPS (Indeks Prestasi Semester), "
        "bukan IPK kumulatif. Jika IPS > 3.0, maksimal SKS yang bisa diambil adalah 24 SKS. "
        "Jika IPS <= 3.0, maksimal adalah 20 SKS. Dengan IPS 2.8, mahasiswa hanya dapat "
        "mengambil maksimal 20 SKS."
    ),
    "d2-08": (
        "Predikat Cum Laude diberikan kepada lulusan dengan IPK 3.51-4.00 dengan syarat "
        "tambahan masa studi tidak melebihi 5 tahun. Jika masa studi lebih dari 5 tahun "
        "meskipun IPK >= 3.51, predikat Cum Laude tidak diberikan. Predikat lainnya: "
        "IPK 3.01-3.50 = Sangat Memuaskan; IPK 2.75-3.00 = Memuaskan. Untuk IPK di bawah "
        "2.75, tabel yudisium tidak mencantumkan label predikatnya."
    ),
    "d2-05": (
        "Syarat umum MBKM di FT Unpas: (1) sudah melewati semester 4, (2) SKS lulus minimal "
        "70 SKS, (3) IPK minimal 3.0, (4) akumulasi konversi MBKM sebelumnya < 30 SKS. "
        "Dengan IPK 2.9, mahasiswa TIDAK memenuhi syarat IPK minimal 3.0, sehingga belum "
        "bisa mendaftar MBKM meskipun sudah di semester 4."
    ),
    "d2-15": (
        "Mahasiswa semester 6 dengan IPK 3.2 dan SKS lulus 95 memenuhi syarat MBKM "
        "(sudah melewati semester 4, IPK >= 3.0, SKS lulus >= 70 SKS). Untuk mengambil "
        "MBKM sambil tetap ambil KRS reguler, mahasiswa bisa memilih skema Partially MBKM. "
        "Dalam skema ini, total SKS (konversi MBKM + KRS reguler) tidak boleh melebihi "
        "24 SKS per semester, dan harus mendapat persetujuan Pimpinan Prodi."
    ),
}

EXTRA_CORRECTIONS = {
    "d1-26": (
        "Evaluasi kinerja akademik Tahap I berlaku bagi mahasiswa pada akhir Tahun Akademik II "
        "dengan IPK <= 2.0 dan SKS lulus <= 40. Agar tidak terkena evaluasi, mahasiswa harus "
        "memiliki IPK di atas 2.0 dan SKS lulus di atas 40."
    ),
    "d2-32": (
        "Berdasarkan Tata Tertib Mahasiswa UNPAS, menyontek saat ujian termasuk pelanggaran "
        "ringan. Sanksi ringan meliputi teguran lisan/tulisan, ganti rugi, dikeluarkan dari "
        "kegiatan kuliah atau ujian, atau tidak diberikan pelayanan administrasi dan akademik "
        "selama 1 bulan. Yang berwenang menjatuhkan sanksi ringan adalah Ketua Jurusan/Prodi, "
        "dosen, atau pegawai/karyawan."
    ),
}


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        correction = CORRECTIONS.get(row["id"])
        if correction:
            row["reference"] = correction
            row["metadata"]["reference_corrected"] = True
    TARGET_V1.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} samples to {TARGET_V1} ({len(CORRECTIONS)} references corrected)")

    for row in rows:
        correction = EXTRA_CORRECTIONS.get(row["id"])
        if correction:
            row["reference"] = correction
            row["metadata"]["reference_corrected"] = True
    TARGET_V2.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    corrected = len(CORRECTIONS) + len(EXTRA_CORRECTIONS)
    print(f"wrote {len(rows)} samples to {TARGET_V2} ({corrected} references corrected)")


if __name__ == "__main__":
    main()
