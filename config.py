"""
@author: Jiri Martinek

Global configuration.
Class is used as section.

Example usage:
from settings import config
config.paths.root
"""

import os

WANDB_DIR = "wandb"
WANDB_API_KEY_DIR = "wandb_private/wandbkey.txt"

class paths:
    root = __file__[:-len("config.py")]    # directory path
    config = __file__

    bin = os.path.join(root, "bin")
    data = os.path.join(root, "data")
    log = os.path.join(root, "log")

    train_bin = os.path.join(bin, "chart_type_image_experiment")

    logger_config = os.path.join(root, "settings", "logger.conf")
    logger_logfile = os.path.join(log, "logger.log")

    dataset_train = os.path.join(data, "icpr2022_chart_image_dataset", "train")
    dataset_test = os.path.join(data, "icpr2022_chart_image_dataset", "test")
    # dataset_val = os.path.join(data, "icpr2022_chart_image_dataset", "dev")

class wandb:
    project = "chart_type_image_experiment"
    entity = "jimar"

class dataset_info:
    chart_type_labels = [
        "area",
        "heatmap",
        "horizontal_bar",
        "horizontal_interval",
        "line",
        "manhattan",
        "map",
        "pie",
        "scatter",
        "scatter-line",
        "surface",
        "venn",
        "vertical_bar",
        "vertical_box",
        "vertical_interval"
    ]

    # only four categories we are interested in (I consider the hor and the ver bar ==> as one category)
    desired_categories: [] = ["horizontal_bar", "vertical_bar", "scatter", "line"]


