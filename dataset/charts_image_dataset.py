# import librosa
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import config
from rich import print
import matplotlib.pyplot as plt
import seaborn as sbs
import cv2

desired_categories: [] = ["horizontal_bar", "vertical_bar", "scatter", "line", "pie"]
USE_ONLY_FOUR_CATEGORIES = False


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
        if USE_ONLY_FOUR_CATEGORIES and chart_type_folder not in desired_categories:
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


class ChartImageTypeDataset(Dataset):
    def __init__(self, dataset_part: str) -> None:
        super().__init__()
        assert dataset_part in ["test", "val", "dev", "train"]
        if dataset_part == "train":
            files: dict[str, str] = prepare_train_data(config.paths.dataset_train)
        elif dataset_part == "test":
            files: dict[str, str] = prepare_test_data(config.paths.dataset_test)
        else:
            files: dict[str, str] = {}

        fixed_width: int = 500
        fixed_height: int = 500

        # let's make two lists
        x = []
        y = []
        for img_file, json_anotation_file in tqdm(files.items(), "Loading images"):
            # input
            # 1) first load the image
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            # 2) resize to a common width, height
            img = cv2.resize(img, (fixed_width, fixed_height))
            # 3) conversion to torch.tensor and add it to the x list
            input_tensor = torch.from_numpy(img)
            # 4) we must rearrange the order (width, height, channel) --> (channel, width, height)
            input_tensor = torch.permute(input_tensor, (2, 0, 1))
            x.append(input_tensor)

            # 4) load json for annotation and take the chart type label
            with open(json_anotation_file, mode="r", encoding="utf8") as fr:
                json_data = json.load(fr)
            label: str = json_data["task1"]["output"]["chart_type"]
            # possible error in the data, vertical bar instead of vertical_bar
            if " " in label:
                label = label.replace(" ", "_")
            class_id: int = config.dataset_info.chart_type_labels.index(label)
            y.append(class_id)
            # for debugging, just 100 samples (for quicker loading)
            # if len(y) == 100:
            #    break
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

    def get_input_len(self):
        return len(self.x[0])


if __name__ == '__main__':
    ds = ChartImageTypeDataset(dataset_part="train")
    for x, y in zip(ds.x, ds.y):
        print(x.shape)
        print(x)
        print(x.shape)
        print(y)
        break
