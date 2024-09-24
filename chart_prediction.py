import argparse

import torch
import sys, os
from model import create_aru_net
from utils import(
    evaluate,
    load_checkpoint,
    get_test_loaders,
    save_test_predictions_as_imgs,
)

# Params
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
# Set the correct parameters according to the model
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PADDING = False # True when the model was trained with data augmentation / random downsampling
PIN_MEMORY = True

# ARU-NET Params
SCALE_SPACE_NUM = 6
RES_DEPTH = 3
FEAT_ROOT = 8 # starting root for features
FILTER_SIZE = 3 # size of kernel
POOL_SIZE = 2 # size of pooling
ACTIVATION_NAME = "relu" # choose "relu" or "elu"
MODEL = "u" # choose "aru", "ru", or "u"
NUM_SCALES = 5 # amount of scaled images you want to use you (e.g. 3: original image and two downscaled versions)

# Models

AXIS_SEGMENTATION_MODEL = os.path.join("final_models", "axis_segmentation" ,"my_checkpoint_best_dice_score.pth.tar")
DATA_VALUES_LINES_SEGMENTATION_MODEL = os.path.join("final_models", "data_values_segmentation_lines" ,"my_checkpoint_best_dice_score.pth.tar")
DATA_VALUES_SCATTER_POINTS_SEGMENTATION_MODEL = os.path.join("final_models", "data_values_segmentation_scatter_points" ,"my_checkpoint_best_dice_score.pth.tar")
DATA_VALUES_VERTICAL_BARS_SEGMENTATION_MODEL = os.path.join("final_models", "data_values_segmentation_vertical_bars" ,"my_checkpoint_best_dice_score.pth.tar")
DATA_VALUES_HORIZONTAL_BARS_SEGMENTATION_MODEL = os.path.join("final_models", "data_values_segmentation_horizontal_bars" ,"my_checkpoint_best_dice_score.pth.tar")

u_net_model_list = [
    AXIS_SEGMENTATION_MODEL,
    DATA_VALUES_LINES_SEGMENTATION_MODEL,
    DATA_VALUES_SCATTER_POINTS_SEGMENTATION_MODEL,
    DATA_VALUES_VERTICAL_BARS_SEGMENTATION_MODEL,
    DATA_VALUES_HORIZONTAL_BARS_SEGMENTATION_MODEL
]



OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



def main(args):
    print(f"Predicting folder: {args.input_folder}")
    TEST_IMG_DIR = args.input_folder
    if TEST_IMG_DIR[-1] != "/":
        TEST_IMG_DIR = TEST_IMG_DIR + "/"

    threshold = args.threshold
    if threshold is not None:
        print(f"Threshold = {threshold}")
        if not (0.0 < threshold < 1.0):
            print(f"Bad threshold {threshold}. Use only number between 0.0 and 1.0")
            threshold = None

    model_kwargs = dict(
        scale_space_num = SCALE_SPACE_NUM,
        res_depth = RES_DEPTH,
        feat_root = FEAT_ROOT,
        filter_size = FILTER_SIZE,
        pool_size = POOL_SIZE,
        activation_name = ACTIVATION_NAME,
        model = MODEL,
        num_scales = NUM_SCALES,
    )

    model = create_aru_net(in_channels = 3, out_channels=1, model_kwargs = model_kwargs).to(DEVICE)
    test_loader = get_test_loaders(TEST_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH, PADDING, NUM_WORKERS, PIN_MEMORY)

    # predictions according to the params given
    if args.enable_axis_prediction:
        load_checkpoint(torch.load(AXIS_SEGMENTATION_MODEL), model)
        output = os.path.join(OUTPUT_DIR, "axis")
        if not os.path.exists(output):
            os.makedirs(output)
        save_test_predictions_as_imgs(test_loader, model, image_height = IMAGE_HEIGHT, image_width = IMAGE_WIDTH, padding = PADDING, output_dir=output, device=DEVICE, threshold=threshold)

    if args.enable_vertical_bars_prediction:
        load_checkpoint(torch.load(DATA_VALUES_VERTICAL_BARS_SEGMENTATION_MODEL), model)
        output = os.path.join(OUTPUT_DIR, "vertical_bars")
        if not os.path.exists(output):
            os.makedirs(output)
        save_test_predictions_as_imgs(test_loader, model, image_height = IMAGE_HEIGHT, image_width = IMAGE_WIDTH, padding = PADDING, output_dir=output, device=DEVICE, threshold=threshold)

    if args.enable_horizontal_bars_prediction:
        load_checkpoint(torch.load(DATA_VALUES_HORIZONTAL_BARS_SEGMENTATION_MODEL), model)
        output = os.path.join(OUTPUT_DIR, "horizontal_bars")
        if not os.path.exists(output):
            os.makedirs(output)
        save_test_predictions_as_imgs(test_loader, model, image_height = IMAGE_HEIGHT, image_width = IMAGE_WIDTH, padding = PADDING, output_dir=output, device=DEVICE, threshold=threshold)

    if args.enable_scatter_points_prediction:
        load_checkpoint(torch.load(DATA_VALUES_SCATTER_POINTS_SEGMENTATION_MODEL), model)
        output = os.path.join(OUTPUT_DIR, "scatter_points")
        if not os.path.exists(output):
            os.makedirs(output)
        save_test_predictions_as_imgs(test_loader, model, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH,
                                      padding=PADDING, output_dir=output, device=DEVICE, threshold=threshold)

    if args.enable_lines_prediction:
        load_checkpoint(torch.load(DATA_VALUES_LINES_SEGMENTATION_MODEL), model)
        output = os.path.join(OUTPUT_DIR, "lines")
        if not os.path.exists(output):
            os.makedirs(output)
        save_test_predictions_as_imgs(test_loader, model, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH,
                                      padding=PADDING, output_dir=output, device=DEVICE, threshold=threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text only experiments for multi-class, multi-label or binary classification')
    parser.add_argument('--input_folder', type=str, help='Input Folder with test chart images')
    parser.add_argument('--enable_axis_prediction', action='store_true', help='enable axis prediction')
    parser.add_argument('--enable_vertical_bars_prediction', action='store_true', help='enable vertical bars prediction')
    parser.add_argument('--enable_horizontal_bars_prediction', action='store_true', help='enable horizontal bars prediction')
    parser.add_argument('--enable_scatter_points_prediction', action='store_true', help='enable scatter points prediction')
    parser.add_argument('--enable_lines_prediction', action='store_true', help='enable lines prediction')

    parser.add_argument("--threshold", type=float, default=None, help="Threshold value between 0 a 1")

    args = parser.parse_args()
    main(args)