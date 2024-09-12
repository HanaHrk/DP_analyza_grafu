import cv2
import numpy as np
import os


def detect_axes(img_file):

    # Načtení obrázku
    image = cv2.imread(img_file, cv2.IMREAD_COLOR)
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## threshold is experimentally determined --> the goal is to keep only black lines, and get rid of every colored lines
    # threshold, bin_img = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("binarized_image.png", bin_img)

    # Use canny edge detection
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)

    # smaller_kernel = np.ones((2,2), np.uint8)
    kernel = np.ones((7, 1), np.uint8)
    # img_erosion = cv2.erode(edges, smaller_kernel, iterations=1)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)

    cv2.imshow('edges ', edges)
    cv2.imshow('Dilation', img_dilation)


    # Apply HoughLinesP method to
    # to directly obtain line end points

    min_line_length = int(image.shape[0] * 0.5) # at least 50% of image widht or height

    lines_list = []
    lines = cv2.HoughLinesP(
        img_dilation,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=50,  # Min number of votes for valid line
        minLineLength=min_line_length,  # Min allowed length of line
        maxLineGap=10  # Max allowed gap between line for joining them
    )

    if lines is not None:
        # Iterate over points
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1, y1), (x2, y2)])
    else:
        print("No lines detected.")

    # Save the result image
    cv2.imshow('Detected Axes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def analyze_imgs(img_folder):
    for file in os.listdir(img_folder):
        print(os.path.join(img_folder, file))
        detect_axes(os.path.join(img_folder, file))


img_folder = "data/black_and_white_imgs"
# img_folder = "data/colored_imgs"
analyze_imgs(img_folder)