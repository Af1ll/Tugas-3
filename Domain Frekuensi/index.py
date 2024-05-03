import cv2
import numpy as np
from matplotlib import pyplot as plt

# Fungsi untuk mengubah citra menjadi grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Fungsi untuk menghitung DFT (Discrete Fourier Transform) dari citra
def compute_dft(image):
    return np.fft.fftshift(np.fft.fft2(image))

# Fungsi untuk menghitung DFT invers dari citra
def compute_inverse_dft(dft_image):
    return np.fft.ifft2(np.fft.ifftshift(dft_image)).real

# Fungsi untuk menerapkan filter frekuensi rendah pada citra dalam domain frekuensi
def apply_low_pass_filter(dft_image, threshold):
    rows, cols = dft_image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[center_row - threshold:center_row + threshold, center_col - threshold:center_col + threshold] = 1
    filtered_dft_image = dft_image * mask
    return filtered_dft_image

# Fungsi untuk menerapkan filter frekuensi tinggi pada citra dalam domain frekuensi
def apply_high_pass_filter(dft_image, threshold):
    rows, cols = dft_image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[center_row - threshold:center_row + threshold, center_col - threshold:center_col + threshold] = 0
    filtered_dft_image = dft_image * mask
    return filtered_dft_image

# Fungsi untuk menerapkan filter Laplacian pada citra dalam domain frekuensi
def apply_laplacian_filter(dft_image):
    rows, cols = dft_image.shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[center_row - 1:center_row + 2, center_col - 1:center_col + 2] = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    filtered_dft_image = dft_image * mask
    return filtered_dft_image

# Load citra
image = cv2.imread('G:\MK\SEMESTER 4\PCD\Tugas 3\Domain Frekuensi\\3_11zon-1.jpg')

# Mengubah citra menjadi grayscale
gray_image = grayscale(image)

# Menghitung DFT dari citra grayscale
dft_image = compute_dft(gray_image)

# Menerapkan filter frekuensi rendah pada citra dalam domain frekuensi
low_pass_filtered_dft_image = apply_low_pass_filter(dft_image, threshold=50)

# Menerapkan filter frekuensi tinggi pada citra dalam domain frekuensi
high_pass_filtered_dft_image = apply_high_pass_filter(dft_image, threshold=50)

# Menerapkan filter Laplacian pada citra dalam domain frekuensi
laplacian_filtered_dft_image = apply_laplacian_filter(dft_image)

# Menghitung DFT invers dari citra dalam domain frekuensi yang telah difilter
low_pass_filtered_image = compute_inverse_dft(low_pass_filtered_dft_image)
high_pass_filtered_image = compute_inverse_dft(high_pass_filtered_dft_image)
laplacian_filtered_image = compute_inverse_dft(laplacian_filtered_dft_image)

# Menampilkan citra-citra hasil pemrosesan
plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(np.log(1 + np.abs(dft_image)), cmap='gray')
plt.title('DFT of Image')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(low_pass_filtered_image, cmap='gray')
plt.title('Low Pass Filtered Image')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(high_pass_filtered_image, cmap='gray')
plt.title('High Pass Filtered Image')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(laplacian_filtered_image, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(np.abs(low_pass_filtered_dft_image), cmap='gray')
plt.title('Low Pass Filtered DFT')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(np.abs(high_pass_filtered_dft_image), cmap='gray')
plt.title('High Pass Filtered DFT')
plt.axis('off')

plt.show()

