
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path

# ==========================================
# 1. MEMBACA GAMBAR
# ==========================================

daftar_gambar = ["gambar.jpg", "gambar (2).jpg"]


def proses_image_enhancement(nama_file):
    img = iio.imread(nama_file)

    # Mengubah RGB menjadi grayscale
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]

        gray = (
            0.299 * img[:, :, 0]
            + 0.587 * img[:, :, 1]
            + 0.114 * img[:, :, 2]
        )
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        gray = img.astype(np.uint8)

    # ==========================================
    # 2. BRIGHTENING (PENINGKATAN KECERAHAN)
    # ==========================================
    bright = np.clip(
        gray.astype(np.int16) + 50, 0, 255
    ).astype(np.uint8)

    # ==========================================
    # 3. IMAGE NEGATIVE (CITRA NEGATIF)
    # ==========================================
    negative = 255 - gray

    # ==========================================
    # 4. CONTRAST STRETCHING
    # ==========================================
    min_val = int(gray.min())
    max_val = int(gray.max())

    if max_val > min_val:
        contrast = (
            (gray.astype(np.float32) - min_val)
            * 255.0 / (max_val - min_val)
        ).clip(0, 255).astype(np.uint8)
    else:
        contrast = gray.copy()

    # ==========================================
    # 5. HISTOGRAM EQUALIZATION
    # ==========================================
    histogram = np.bincount(
        gray.ravel(), minlength=256
    )
    cdf = histogram.cumsum()
    cdf_min = cdf[cdf > 0][0]

    if gray.size > cdf_min:
        equalized = np.interp(
            gray.ravel(),
            np.arange(256),
            (cdf - cdf_min) * 255.0 / (gray.size - cdf_min)
        ).clip(0, 255).astype(np.uint8).reshape(gray.shape)
    else:
        equalized = gray.copy()

    # ==========================================
    # 6. MEDIAN FILTER (PENGURANGAN NOISE)
    # ==========================================
    denoised = ndimage.median_filter(gray, size=3)

    # ==========================================
    # 7. MENYIMPAN HASIL
    # ==========================================
    nama_dasar = Path(nama_file).stem

    iio.imwrite(f"{nama_dasar}_grayscale.png", gray)
    iio.imwrite(f"{nama_dasar}_brightening.png", bright)
    iio.imwrite(f"{nama_dasar}_negative.png", negative)
    iio.imwrite(f"{nama_dasar}_contrast.png", contrast)
    iio.imwrite(f"{nama_dasar}_equalization.png", equalized)
    iio.imwrite(f"{nama_dasar}_median_filter.png", denoised)

    # ==========================================
    # 8. MENAMPILKAN PERBANDINGAN
    # ==========================================
    hasil = [
        (gray, "Citra Grayscale"),
        (bright, "Brightening"),
        (negative, "Image Negative"),
        (contrast, "Contrast Stretching"),
        (equalized, "Histogram Equalization"),
        (denoised, "Median Filter")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for ax, (gambar, judul) in zip(axes.ravel(), hasil):
        ax.imshow(gambar, cmap="gray", vmin=0, vmax=255)
        ax.set_title(judul)
        ax.axis("off")

    fig.suptitle(f"Image Enhancement: {nama_file}", fontsize=14)
    plt.tight_layout()

    nama_hasil = f"{nama_dasar}_hasil_enhancement.png"
    plt.savefig(nama_hasil, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"\nGambar diproses: {nama_file}")
    print("Ukuran gambar:", gray.shape)
    print("Intensitas minimum:", int(gray.min()))
    print("Intensitas maksimum:", int(gray.max()))
    print("Hasil disimpan:", nama_hasil)


# ==========================================
# 9. MEMPROSES SEMUA GAMBAR
# ==========================================

for nama_file in daftar_gambar:
    try:
        proses_image_enhancement(nama_file)
    except FileNotFoundError:
        print(f"File tidak ditemukan: {nama_file}")

print("\nSemua gambar yang tersedia selesai diproses.")
