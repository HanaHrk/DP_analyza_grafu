# DP Analýza grafů

## README for this project

## U-Net models for various segmentation tasks
There are five models in the ´final models´ folder.
- axis segmentation
- vertical bars segmentation
- horizontal bars segmentation
- scatter points segmentation
- lines segmentation
The model is called U-net [https://arxiv.org/pdf/1505.04597], and it is a representative of Fully Convolutional Network. The model is learned to predict segmentation mask.

### Usage
Use the ´chart_prediction.py´ script. The scripts takes a folder as an input and for each image within calls desired models for predictions. E.g. ´python chart_prediction.py --input_folder sample_images/vertical_bars/ --enable_vertical_bars_prediction´, this configuration goes through the ´sample_images/vertical_bars´ folder and creates an output folder ´output/vertical_bars´. In this directory, there are vertical bar predictions.

For any other configuration see the script and its params.
