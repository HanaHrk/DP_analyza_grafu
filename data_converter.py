import shutil
from tqdm import tqdm
import os
from rich import print
import config
import json
import cv2

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


def create_gt_mask_axis(img_file: str, json_file: str):
    # 1) load image and create an empty image with lines as background
    orig_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    h, w = orig_img.shape
    axis_gt_img = orig_img.copy()
    axis_gt_img[:, :] = 0  # create black img with the same size

    # 2) load annotation in json
    with open(json_file, mode="r", encoding="utf8") as fr:
        json_data = json.load(fr)

    if len(json_data["task4"]["output"]["axes"]["x-axis"]) > 0:
        x_axis_1_beginning_point_x: str = json_data["task4"]["output"]["axes"]["x-axis"][0]["tick_pt"]["x"]
        x_axis_1_beginning_point_y: str = json_data["task4"]["output"]["axes"]["x-axis"][0]["tick_pt"]["y"]
        x_axis_1_end_point_x: str = json_data["task4"]["output"]["axes"]["x-axis"][-1]["tick_pt"]["x"]
        x_axis_1_end_point_y: str = json_data["task4"]["output"]["axes"]["x-axis"][-1]["tick_pt"]["y"]

        point1 = (int(x_axis_1_beginning_point_x), int(x_axis_1_beginning_point_y))
        point2 = (int(x_axis_1_end_point_x), int(x_axis_1_end_point_y))
        axis_gt_img = cv2.line(axis_gt_img, point1, point2, color=(255, 255, 255), thickness=10)

    if len(json_data["task4"]["output"]["axes"]["x-axis-2"]) > 0:
        x_axis_2_beginning_point_x: str = json_data["task4"]["output"]["axes"]["x-axis-2"][0]["tick_pt"]["x"]
        x_axis_2_beginning_point_y: str = json_data["task4"]["output"]["axes"]["x-axis-2"][0]["tick_pt"]["y"]
        x_axis_2_end_point_x: str = json_data["task4"]["output"]["axes"]["x-axis-2"][-1]["tick_pt"]["x"]
        x_axis_2_end_point_y: str = json_data["task4"]["output"]["axes"]["x-axis-2"][-1]["tick_pt"]["y"]

        point1 = int(x_axis_2_beginning_point_x), int(x_axis_2_beginning_point_y)
        point2 = int(x_axis_2_end_point_x), int(x_axis_2_end_point_y)
        axis_gt_img = cv2.line(axis_gt_img, point1, point2, color=(255, 255, 255), thickness=10)

    if len(json_data["task4"]["output"]["axes"]["y-axis"]) > 0:
        y_axis_1_beginning_point_x: str = json_data["task4"]["output"]["axes"]["y-axis"][0]["tick_pt"]["x"]
        y_axis_1_beginning_point_y: str = json_data["task4"]["output"]["axes"]["y-axis"][0]["tick_pt"]["y"]
        y_axis_1_end_point_x: str = json_data["task4"]["output"]["axes"]["y-axis"][-1]["tick_pt"]["x"]
        y_axis_1_end_point_y: str = json_data["task4"]["output"]["axes"]["y-axis"][-1]["tick_pt"]["y"]

        point1 = (int(y_axis_1_beginning_point_x), int(y_axis_1_beginning_point_y))
        point2 = (int(y_axis_1_end_point_x), int(y_axis_1_end_point_y))
        axis_gt_img = cv2.line(axis_gt_img, point1, point2, color=(255, 255, 255), thickness=10)

    if len(json_data["task4"]["output"]["axes"]["y-axis-2"]) > 0:
        y_axis_2_beginning_point_x: str = json_data["task4"]["output"]["axes"]["y-axis-2"][0]["tick_pt"]["x"]
        y_axis_2_beginning_point_y: str = json_data["task4"]["output"]["axes"]["y-axis-2"][0]["tick_pt"]["y"]
        y_axis_2_end_point_x: str = json_data["task4"]["output"]["axes"]["y-axis-2"][-1]["tick_pt"]["x"]
        y_axis_2_end_point_y: str = json_data["task4"]["output"]["axes"]["y-axis-2"][-1]["tick_pt"]["y"]

        point1 = int(y_axis_2_beginning_point_x), int(y_axis_2_beginning_point_y)
        point2 = int(y_axis_2_end_point_x), int(y_axis_2_end_point_y)
        axis_gt_img = cv2.line(axis_gt_img, point1, point2, color=(255, 255, 255), thickness=10)

    output_filepath = os.path.join(config.paths.data, "all_masks", os.path.basename(img_file))
    cv2.imwrite(output_filepath, axis_gt_img)

    # copy original img
    shutil.copy(img_file, os.path.join(config.paths.data, "all_images"))


def filter_task_4_data(img2json_train) -> dict[str, str]:
    filtered_img2json_train: dict[str, str] = {}

    print("Filtering data")

    # we must load each json file and check if it contains task_6 data (data_extraction information)
    for img_file, json_file in img2json_train.items():
        with open(json_file, mode="r", encoding="utf8") as fr:
            json_data = json.load(fr)
        # print(json_file)
        if "task4" not in json_data or json_data["task4"] is None:
            continue
        else:
            filtered_img2json_train[img_file] = json_file

    print(f"Filtered data: {len(filtered_img2json_train)}") ## TOTAL 5028 data for data extraction

    return filtered_img2json_train


if __name__ == '__main__':
    img2json_train: dict[str, str] = prepare_train_data(config.paths.dataset_train)
    print(f"All data: {len(img2json_train)}")
    filtered_img2json_train: dict[str, str] = filter_task_4_data(img2json_train)
    for img_file, json_file in filtered_img2json_train.items():
        create_gt_mask_axis(img_file, json_file)
