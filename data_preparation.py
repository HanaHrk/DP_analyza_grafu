from tqdm import tqdm
import os
from rich import print
import config
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def prepare_test_data(folder):
    print(f"Preparing test data from {os.path.abspath(folder)}")
    # create a dictionary, the key is img filepath and value an apropriate json filepath
    img2json_dict: dict[str, str] = {}
    img_folder = os.path.join(folder, "chart_images", "split_1", "images")
    # json_folder = os.path.join(folder, "annotations_JSON")
    for img_file in tqdm(os.listdir(img_folder), desc="Preparing data"):
        if img_file.endswith(".jpg"):
            # find an apropriate JSON file in folder
            # replace "images" -> annotations_JSON in the abs path
            json_file = os.path.abspath(os.path.abspath(os.path.join(img_folder, img_file))).replace("chart_images", "final_full_GT").replace("images", "annotations_JSON").replace(".jpg", ".json")
            if img_file not in img2json_dict:
                img2json_dict[os.path.abspath(os.path.join(img_folder, img_file))] = json_file

    return img2json_dict

def prepare_train_data(folder):
    print(f"Preparing train data from {os.path.abspath(folder)}")
    # create a dictionary, the key is img filepath and value an apropriate json filepath
    img2json_dict: dict[str, str] = {}
    img_folder = os.path.join(folder, "images")
    # json_folder = os.path.join(folder, "annotations_JSON")
    for chart_type_folder in tqdm(os.listdir(img_folder), desc="Preparing data"):
        if chart_type_folder not in config.dataset_info.desired_categories:
            continue
        else:
            for img_file in os.listdir(os.path.join(img_folder, chart_type_folder)):
                if img_file.endswith(".jpg"):
                    # find an apropriate JSON file in folder
                    # replace "images" -> annotations_JSON in the abs path
                    json_file = os.path.abspath(os.path.join(img_folder, chart_type_folder, img_file)).replace("images", "annotations_JSON").replace(".jpg", ".json")
                    if img_file not in img2json_dict:
                        img2json_dict[os.path.abspath(os.path.join(img_folder, chart_type_folder, img_file))] = json_file

    return img2json_dict


def create_gt_mask_data_extraction(img_file: str, json_file: str):
    with open(json_file, mode="r", encoding="utf8") as fr:
        json_data = json.load(fr)
    label: str = json_data["task1"]["output"]["chart_type"]
    # possible error in the data, vertical bar instead of vertical_bar
    if " " in label:
        label = label.replace(" ", "_")

    if label == "vertical_bar" or label == "horizontal_bar":
        data: [] = json_data["task6"]["visual elements"]["bars"]
    elif label == "line":
        data: [] = json_data["task6"]["visual elements"]["lines"]
    elif label == "scatter":
        data: [] = json_data["task6"]["visual elements"]["scatter points"]

    fixed_width: int = 500
    fixed_height: int = 500

    # 1) load image an

def filter_task_6_data(img2json_train) -> dict[str, str]:
    filtered_img2json_train: dict[str, str] = {}

    print("Filtering data")

    # we must load each json file and check if it contains task_6 data (data_extraction information)
    for img_file, json_file in img2json_train.items():
        with open(json_file, mode="r", encoding="utf8") as fr:
            json_data = json.load(fr)
        # print(json_file)
        if "task6" not in json_data or json_data["task6"] is None:
            continue
        else:
            filtered_img2json_train[img_file] = json_file

    print(f"Filtered data: {len(filtered_img2json_train)}") ## TOTAL 3844 data for data extraction

    # we must find max width and max height
    max_width = 0
    max_height = 0
    widths, heights = [], []
    for img_file in filtered_img2json_train.keys():
        with Image.open(img_file) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height
    print(f"Max width = {max_width}")
    print(f"Max height = {max_height}")

    plt.boxplot(np.array(widths))
    plt.show()
    plt.boxplot(np.array(heights))
    plt.show()
    return filtered_img2json_train


if __name__ == '__main__':
    img2json_train: dict[str, str] = prepare_train_data(config.paths.dataset_train)
    print(f"All data: {len(img2json_train)}")
    filtered_img2json_train: dict[str, str] = filter_task_6_data(img2json_train)
    exit(0)
    for img_file, json_file in filtered_img2json_train.items():
        create_gt_mask_data_extraction(img_file, json_file)