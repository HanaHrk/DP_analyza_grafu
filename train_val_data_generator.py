import os
import config
import random
import shutil
seed = 1403
random.seed(seed)

"""
val_data_ratio == how big the val data is --> 0.1 means 10% of all data = val, 90% = train 
"""
def generate_train_val_data_list(val_data_ratio: float=0.1):
    all_file_list: [] = os.listdir(os.path.join(config.paths.data, "img_ground_truth_masks"))
    # shuffle list --> with seed to be able to reproduce
    random.shuffle(all_file_list)

    # 1-val_data_ratio
    train_data_size = int(len(all_file_list) * (1 - val_data_ratio))
    val_data_size = len(all_file_list) - train_data_size
    assert train_data_size + val_data_size == len(all_file_list)

    with open("train_data_list.txt", mode="w", encoding="utf8") as fw:
        for i in range(0, train_data_size):
            file = all_file_list[i]
            if "GT0.jpg" in file:
                continue
            else:
                # print(os.path.abspath(file))
                fw.write(f"{os.path.abspath(file)}\n")

    with open("val_data_list.txt", mode="w", encoding="utf8") as fw:
        for i in range(train_data_size, len(all_file_list)):
            file = all_file_list[i]
            if "GT0.jpg" in file:
                continue
            else:
                # print(os.path.abspath(file))
                fw.write(f"{os.path.abspath(file)}\n")


if __name__ == '__main__':
    # generate_train_val_data_list()
    with open("val_data_list.txt", mode="r", encoding="utf8") as fw:
        files_list = fw.readlines()
    for file in files_list:
        file = file.strip()
        file = os.path.join(config.paths.data, "img_ground_truth_masks", os.path.basename(file))
        shutil.copy(file, "data/axis_analysis/val_images")
        mask_file = os.path.join(config.paths.data, "img_ground_truth_masks", os.path.basename(file).replace(".jpg", "_GT0.jpg"))
        shutil.copy(mask_file, "data/axis_analysis/val_masks/"+os.path.basename(file).replace("_GT0.jpg", ".jpg"))

    with open("train_data_list.txt", mode="r", encoding="utf8") as fw:
        files_list = fw.readlines()
    for file in files_list:
        file = file.strip()
        file = os.path.join(config.paths.data, "img_ground_truth_masks", os.path.basename(file))
        shutil.copy(file, "data/axis_analysis/train_images")
        mask_file = os.path.join(config.paths.data, "img_ground_truth_masks", os.path.basename(file).replace(".jpg", "_GT0.jpg"))
        shutil.copy(mask_file, "data/axis_analysis/train_masks/"+os.path.basename(file).replace("_GT0.jpg", ".jpg"))
