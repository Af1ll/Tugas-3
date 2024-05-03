import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk mengubah citra menjadi grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Fungsi untuk menerapkan filter rata-rata pada citra
def average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

# Fungsi untuk menerapkan filter Gaussian pada citra
def gaussian_filter(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Fungsi untuk menerapkan filter median pada citra
def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

# Fungsi untuk memperoleh tepi citra menggunakan operator Sobel
def sobel_edge_detection(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    return gradient_magnitude.astype(np.uint8)

# Fungsi untuk memperoleh tepi citra menggunakan operator Canny
def canny_edge_detection(image, threshold1, threshold2):
    return cv2.Canny(image, threshold1, threshold2)

# Fungsi untuk menerapkan filter sharpening pada citra
def sharpening_filter(image, kernel_size, alpha):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    sharpened = cv2.addWeighted(image, 1 + alpha, blurred, -alpha, 0)
    return sharpened

# Fungsi untuk menerapkan filter min pada citra
def min_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel)

# Fungsi untuk menerapkan filter max pada citra
def max_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel)

# Fungsi untuk menerapkan filter laplacian pada citra
def laplacian_filter(image):
    return cv2.Laplacian(image, cv2.CV_64F)


# Load citra
image = cv2.imread('G:\MK\SEMESTER 4\PCD\Tugas 3\\3_11zon-1.jpg')

# Mengubah citra menjadi grayscale
gray_image = grayscale(image)

# Menerapkan filter rata-rata pada citra grayscale
average_filtered_image = average_filter(gray_image, kernel_size=5)

# Menerapkan filter Gaussian pada citra grayscale
gaussian_filtered_image = gaussian_filter(gray_image, kernel_size=5)

# Menerapkan filter median pada citra grayscale
median_filtered_image = median_filter(gray_image, kernel_size=5)

# Mendeteksi tepi citra menggunakan operator Sobel pada citra grayscale
sobel_edges = sobel_edge_detection(gray_image)

# Mendeteksi tepi citra menggunakan operator Canny pada citra grayscale
canny_edges = canny_edge_detection(gray_image, threshold1=100, threshold2=200)

# Menerapkan filter sharpening pada citra grayscale
sharpened_image = sharpening_filter(gray_image, kernel_size=5, alpha=1.0)

# Menerapkan filter min pada citra grayscale
min_filtered_image = min_filter(image, kernel_size=3)

# Menerapkan filter max pada citra grayscale
max_filtered_image = max_filter(image, kernel_size=3)

# Menerapkan filter laplacian pada citra grayscale
laplacian_filtered_image = laplacian_filter(image)

# Menampilkan citra-citra hasil pemrosesan
plt.subplot(4, 3, 1)
plt.imshow(image[:,:,::-1])
plt.title('Original Image')
plt.axis('off')

plt.subplot(4, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Scale Image')
plt.axis('off')

plt.subplot(4, 3, 3)
plt.imshow(average_filtered_image, cmap='gray')
plt.title('Average Filtered Image')
plt.axis('off')

plt.subplot(4, 3, 4)
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.title('Gaussian Filtered Image')
plt.axis('off')

plt.subplot(4, 3, 5)
plt.imshow(median_filtered_image, cmap='gray')
plt.title('Median Filtered Image')
plt.axis('off')

plt.subplot(4, 3, 6)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Sobel Edges Image')
plt.axis('off')

plt.subplot(4, 3, 7)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges Image')
plt.axis('off')

plt.subplot(4, 3, 8)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.subplot(4, 3, 9)
plt.imshow(min_filtered_image, cmap='gray')
plt.title('Min filter Image')
plt.axis('off')

plt.subplot(4, 3, 10)
plt.imshow(max_filtered_image, cmap='gray')
plt.title('Max Filter Image')
plt.axis('off')

plt.subplot(4, 3, 11)
plt.imshow(laplacian_filtered_image, cmap='gray')
plt.title('Laplacian Filutered Image')
plt.axis('off')

plt.show()


