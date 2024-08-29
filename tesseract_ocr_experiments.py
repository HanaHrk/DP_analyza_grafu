
import json

import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw

def draw_tesseract_output(data, img):
    # drawing detected boudning boxes and texts to the image
    draw = ImageDraw.Draw(img)

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 75:  # confidence at least 75
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            text = data['text'][i]
            print(f"Text: {text}, Position: ({x}, {y}), Size: ({w}x{h}), Confidence: {data['conf'][i]}")

            draw.rectangle([x, y, x + w, y + h], outline='red', width=2)

            draw.text((x, y - 10), text, fill='red')
    output_image_path = 'ocr_output/tesseract_detection.png'
    img.save(output_image_path)


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
text_blocks = json_data["task2"]["output"]["text_blocks"]

# loading image with Pillow
img = Image.open(image_path)
draw_json_annotation(text_blocks, img)


# loading image with Pillow
img = Image.open(image_path)

# pyttesseract for OCR with an output including bounding boxes positions
data = pytesseract.image_to_data(img, output_type=Output.DICT)




draw_tesseract_output(data, img)




