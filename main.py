
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# 1. MEMBACA GAMBAR ASLI (RGB)
# =========================================

nama_file = "gambar.jpg"

img = iio.imread(nama_file)

if img.ndim != 3 or img.shape[2] < 3:
    raise ValueError("Gunakan gambar berwarna RGB.")

original = img[:, :, :3]

# =========================================
# 2. MENGUBAH RGB MENJADI GRAYSCALE
# =========================================

rgb = original.astype(np.float32)

gray = (
    0.299 * rgb[:, :, 0]
    + 0.587 * rgb[:, :, 1]
    + 0.114 * rgb[:, :, 2]
)

gray = np.clip(gray, 0, 255).astype(np.uint8)

# =========================================
# 3. MENGHITUNG HISTOGRAM GRAYSCALE
# =========================================

histogram = np.bincount(
    gray.ravel(),
    minlength=256
)

intensitas = np.arange(256)

total_piksel = gray.size

intensitas_dominan = np.where(
    histogram == histogram.max()
)[0]

# =========================================
# 4. MENAMPILKAN INFORMASI HASIL
# =========================================

print("===== ANALISIS HISTOGRAM GRAYSCALE =====")
print(f"Nama gambar       : {nama_file}")
print(f"Ukuran gambar     : {gray.shape[1]} x {gray.shape[0]}")
print(f"Jumlah total piksel: {total_piksel}")

print("\nFrekuensi setiap intensitas:")

for i in range(256):
    print(f"Intensitas {i:3d}: {histogram[i]} piksel")

print("\nIntensitas dominan:",
      intensitas_dominan.tolist())

print("Jumlah piksel pada intensitas dominan:",
      histogram.max())

# =========================================
# 5. MENAMPILKAN GAMBAR DAN HISTOGRAM
# =========================================

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Gambar RGB
ax[0].imshow(original)
ax[0].set_title("Gambar Asli (RGB)")
ax[0].axis("off")

# Gambar grayscale
ax[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("Gambar Grayscale")
ax[1].axis("off")

# Histogram grayscale
ax[2].bar(
    intensitas,
    histogram,
    width=1.0,
    color="steelblue"
)

ax[2].set_title("Histogram Grayscale")
ax[2].set_xlabel("Intensitas Piksel (0-255)")
ax[2].set_ylabel("Jumlah Piksel")
ax[2].set_xlim(0, 255)
ax[2].grid(axis="y", alpha=0.3)

plt.tight_layout()

# =========================================
# 6. MENYIMPAN HASIL
# =========================================

plt.savefig(
    "hasil_histogram_grayscale.png",
    dpi=300,
    bbox_inches="tight"
)

iio.imwrite("hasil_grayscale.png", gray)

np.savetxt(
    "frekuensi_histogram.csv",
    np.column_stack((intensitas, histogram)),
    delimiter=",",
    fmt="%d",
    header="Intensitas,Jumlah_Piksel",
    comments=""
)

print("\nFile hasil:")
print("- hasil_histogram_grayscale.png")
print("- hasil_grayscale.png")
print("- frekuensi_histogram.csv")

plt.show()
