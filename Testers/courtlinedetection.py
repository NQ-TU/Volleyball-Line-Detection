import cv2
import numpy as np
from matplotlib import pyplot as plt
import easygui

# Load the input image
f = easygui.fileopenbox()
img = cv2.imread(f)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale for processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform edge detection using Canny
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use Hough Line Transform to extract lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=50)

# Overlay detected lines onto the original image
img_lines = img_rgb.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw lines in blue
        cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 3)

# Show and save result
plt.imshow(img_lines)
plt.title('Court Lines Overlay')
plt.axis('off')
plt.show()

