import json

import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw

def draw_json_annotation(data, img):
    # drawing detected boudning boxes and texts to the image
    draw = ImageDraw.Draw(img)

    for text_block in data:
        # keys = id, polygon, text
        polygon_data = text_block["polygon"]
        x = int(polygon_data["x0"])
        y = int(polygon_data["y0"])
        w = int(polygon_data["x1"]) - int(polygon_data["x0"])
        h = int(polygon_data["y2"]) - int(polygon_data["y0"])
        text = text_block["text"]
        print(f"Text: {text}, Position: ({x}, {y}), Size: ({w}x{h})")

        draw.rectangle([x, y, x + w, y + h], outline='red', width=2)

        draw.text((x, y - 10), text, fill='red')
    output_image_path = 'ocr_output/json_annotation.png'
    img.save(output_image_path)


# path to an example of image
image_path = "/mnt/data/Corpora/chart_dataset/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images/horizontal_bar/PMC3104217___1471-2458-11-S4-S4-1.jpg"
json_annotation_path = "/mnt/data/Corpora/chart_dataset/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/annotations_JSON/horizontal_bar/PMC3104217___1471-2458-11-S4-S4-1.json"

with open(json_annotation_path, mode="r", encoding="utf8") as fr:
    json_data = json.load(fr)
axes_data = json_data["task4"]["output"]["axes"]

print("Osa x:")
print("Pocatek:")
print(axes_data["x-axis"][0]["tick_pt"]["x"])
print(axes_data["x-axis"][0]["tick_pt"]["y"])
print("Konec:")
print(axes_data["x-axis"][-1]["tick_pt"]["x"])
print(axes_data["x-axis"][-1]["tick_pt"]["y"])

print("Osa x:")
print("Pocatek:")
print(axes_data["y-axis"][0]["tick_pt"]["x"])
print(axes_data["y-axis"][0]["tick_pt"]["y"])
print("Konec:")
print(axes_data["y-axis"][-1]["tick_pt"]["x"])
print(axes_data["y-axis"][-1]["tick_pt"]["y"])
exit(0)
# loading image with Pillow
img = Image.open(image_path)
draw_json_annotation(axes_data, img)
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
import cv2
import numpy as np

# Načtení obrázku
image_path = 'vase_cesta_k_obrazku.png'
img = cv2.imread(image_path)

# Převod do odstínů šedi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detekce hran pomocí Cannyho detektoru
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Aplikace Houghovy transformace pro detekci čar
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Vykreslení detekovaných čar do obrázku
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Vykreslení čáry (červená barva, tloušťka 2 px)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Uložení výsledného obrázku
output_image_path = 'output_lines.png'
cv2.imwrite(output_image_path, img)

print(f"Výsledný obrázek byl uložen jako {output_image_path}")