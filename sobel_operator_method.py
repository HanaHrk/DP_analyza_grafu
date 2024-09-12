import cv2
import numpy as np
import os


def detect_axes(image_path):
    # Načtení obrázku
    img = cv2.imread(image_path)

    # Převod na stupně šedi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplikace Gaussian blur pro snížení šumu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detekce hran pomocí Sobelova operátoru
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)  # Derivace podle osy x
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)  # Derivace podle osy y

    # Práh pro nalezení výrazných hran
    _, binary_x = cv2.threshold(np.abs(sobel_x), 100, 255, cv2.THRESH_BINARY)
    _, binary_y = cv2.threshold(np.abs(sobel_y), 100, 255, cv2.THRESH_BINARY)

    # Konverze na 8bitové obrázky pro zpracování
    binary_x = np.uint8(binary_x)
    binary_y = np.uint8(binary_y)

    # Nalezení potenciálních vodorovných a svislých linií
    kernel = np.ones((1, 50), np.uint8)  # Detekce dlouhých vodorovných linií
    horizontal_lines = cv2.morphologyEx(binary_x, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((50, 1), np.uint8)  # Detekce dlouhých svislých linií
    vertical_lines = cv2.morphologyEx(binary_y, cv2.MORPH_CLOSE, kernel)

    # Sloučení vodorovných a svislých linií zpět do jednoho obrázku
    axes = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Zobrazení nalezených os
    cv2.imshow('Detected Axes', axes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def analyze_imgs(img_folder):
    for file in os.listdir(img_folder):
        detect_axes(os.path.join(img_folder, file))


# img_folder = "data/black_and_white_imgs"
img_folder = "data/colored_imgs"
analyze_imgs(img_folder)