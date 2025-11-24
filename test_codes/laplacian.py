import cv2
import numpy as np
from matplotlib import pyplot as plt

# Görüntüyü gri seviyede oku
img = cv2.imread('data/heisenberg.jpg', cv2.IMREAD_GRAYSCALE)

# Laplacian filtresini uygula
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=1)

# Negatif değerleri 0-255 aralığına getir
laplacian = cv2.convertScaleAbs(laplacian)

# Orijinal ve sonuçları göster
plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title('Orijinal')
plt.subplot(1,2,2), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian Filtre')
plt.show()
