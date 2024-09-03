import cv2
import numpy as np

# Read image
image = cv2.imread("/mnt/data/Corpora/chart_dataset/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images/horizontal_bar/PMC3104217___1471-2458-11-S4-S4-1.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

## threshold is experimentally determined --> the goal is to keep only black lines, and get rid of every colored lines
threshold, bin_img = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
cv2.imwrite("binarized_image.png", bin_img)

# Use canny edge detection
edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)

# Apply HoughLinesP method to
# to directly obtain line end points

min_line_length = int(image.shape[0] * 0.75) # at least 75% of image widht or height

lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=100,  # Min number of votes for valid line
    minLineLength=min_line_length,  # Min allowed length of line
    maxLineGap=10  # Max allowed gap between line for joining them
)

# Iterate over points
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])

# Save the result image
cv2.imwrite('detectedLines.png', image)