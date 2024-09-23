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
    data_values_gt_img = orig_img.copy()
    data_values_gt_img[:, :] = 0  # create black img with the same size

    # 2) load annotation in json
    with open(json_file, mode="r", encoding="utf8") as fr:
        json_data = json.load(fr)

    label: str = json_data["task1"]["output"]["chart_type"]
    # possible error in the data, vertical bar instead of vertical_bar
    if " " in label:
        label = label.replace(" ", "_")

    if label == "vertical_bar" or label == "horizontal_bar":
        data: [] = json_data["task6"]["output"]["visual elements"]["bars"]

        for item in data:
            height = int(item["height"])
            width = int(item["width"])
            x0 = int(item["x0"])
            y0 = int(item["y0"])
            x1 = x0 + width
            y1 = y0 + height

            # draw to the mask --> fill rectangle
            data_values_gt_img = cv2.rectangle(data_values_gt_img, (x0, y0), (x1, y1), (255, 255, 255), -1)

    elif label == "line":
        data: [] = json_data["task6"]["output"]["visual elements"]["lines"]

        for list_of_points in data:
            for i in range(len(list_of_points) - 1):
                point1 = (int(list_of_points[i]["x"]), int(list_of_points[i]["y"]))
                point2 = (int(list_of_points[i+1]["x"]), int(list_of_points[i+1]["y"]))
                data_values_gt_img = cv2.line(data_values_gt_img, point1, point2, color=(255, 255, 255), thickness=1)

    elif label == "scatter":
        data: [] = json_data["task6"]["output"]["visual elements"]["scatter points"]

        for list_of_points in data:
            for point in list_of_points:
                center_coordinates = (int(point["x"]), int(point["y"]))
                # draw to the mask --> fill rectangle
                data_values_gt_img = cv2.circle(data_values_gt_img, center_coordinates, radius=3, color=(255, 255, 255), thickness=-1)

    else:
        data: [] = None

    output_filepath = os.path.join(config.paths.data, "data_values_segmentation", "data_values_all_masks", os.path.basename(img_file))
    cv2.imwrite(output_filepath, data_values_gt_img)

    # copy original img
    shutil.copy(img_file, os.path.join(config.paths.data, "data_values_segmentation", "data_values_all_images/"))


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
            label: str = json_data["task1"]["output"]["chart_type"]
            # possible error in the data, vertical bar instead of vertical_bar
            if " " in label:
                label = label.replace(" ", "_")
            if label == "line":
                filtered_img2json_train[img_file] = json_file

    print(f"Filtered data: {len(filtered_img2json_train)}") ## TOTAL 3844 data for data extraction
    return filtered_img2json_train


if __name__ == '__main__':
    img2json_train: dict[str, str] = prepare_train_data(config.paths.dataset_train)
    print(f"All data: {len(img2json_train)}")
    filtered_img2json_train: dict[str, str] = filter_task_6_data(img2json_train)
    for img_file, json_file in filtered_img2json_train.items():
        create_gt_mask_axis(img_file, json_file)
