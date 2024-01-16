import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import sys
import json
import cv2
import os

# C# tarafından gelen JSON verisini oku
input_data = sys.stdin.readline()
data = json.loads(input_data)

inputImagePath = data["inputImagePath"]
outputImagePath = data["outputImagePath"]

file_name, file_extension = os.path.splitext(os.path.basename(inputImagePath))

# Modeli yükle
model = load_model(r'C:\Users\semih\Desktop\ocrtext\trained_model2.h5')

# Görüntüyü oku
image = cv2.imread(inputImagePath)

# Görüntüyü gri tonlamaya çevir
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gri tonlamalı görüntüyü renkliye çevir
color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# Eşik değerlerini belirle
threshold1 = 70
threshold2 = 100

# Kenarları bul
edges = cv2.Canny(gray_image, threshold1, threshold2)

# Konturları bul
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

detected_letters = []

# Her bir konturun etrafına dikdörtgen çiz ve harf tespiti yap
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    # Her bir kutuyu kırp ve boyutlandır
    letter_roi = cv2.resize(color_image[y:y+h, x:x+w], (64, 64))
    
    # Kırpılan harfi normalleştir
    letter_roi = letter_roi / 255.0
    
    # Boyutu (1, 64, 64, 3) olan bir dizi oluştur
    letter_roi = np.expand_dims(letter_roi, axis=0)
    
    # Harfi modelinize iletip tahmin yapın
    predicted_class = np.argmax(model.predict(letter_roi))
    
    # Örnek: Tanıma sonucunu al
    recognized_letter = chr(ord('A') + predicted_class)
    
    # Her bir harfi ve konumunu listeye ekle
    detected_letters.append((recognized_letter, (x, y, x+w, y+h)))
    
    # Harfi görüntü üzerine yazdır
    cv2.putText(image, recognized_letter, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)  # Mavi renk, küçük font
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)  # Mavi renk

# Görüntüyü Matplotlib ile göster
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Letters')
plt.show()

# Her bir harfi ve konumunu yazdır
for letter, (x1, y1, x2, y2) in detected_letters:
    print(f"Detected {letter} at position ({x1}, {y1}, {x2}, {y2})")

cv2.imwrite(outputImagePath + file_name + "_sonuc.jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


