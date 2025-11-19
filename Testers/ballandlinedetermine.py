import cv2
import numpy as np
from matplotlib import pyplot as plt
import easygui

# Load image
f = easygui.fileopenbox()
img = cv2.imread(f)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to RGB for display
imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Automatic Ball Segmentation with K-Means ---
Z = img.reshape((-1, 3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3  # Can change to 4 if needed
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
seg_img = res.reshape((img.shape))

# Convert clustered image to grayscale and threshold for mask
seg_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
# Use Otsu's thresholding for adaptive mask selection
_, thresh = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Optionally invert mask if needed based on background/ball colors

# Extract and draw ball contour
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgcontours = imgrgb.copy()
ball_center = None
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(imgcontours, [largest_contour], -1, (0, 255, 0), 3)

    # Calculate the center of the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        ball_center = (cX, cY)

# --- Court Line Detection with Canny and Hough Transform ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150, apertureSize=3)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=7)
lineimg = imgcontours.copy()  # Start with ball+contour

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lineimg, (x1, y1), (x2, y2), (255, 0, 0), 4)

# --- In/Out Classification Based on Ball Position ---
def check_ball_in_out(ball_center, lines, threshold=5):
    """
    Checks if the ball has crossed a line.
    """
    if ball_center is None:
        return "No Ball Detected"
    
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate the distance from the ball center to the line
        distance = np.abs((y2 - y1) * ball_center[0] - (x2 - x1) * ball_center[1] + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

        if distance < threshold:
            return "Out"  # Ball is out if it intersects with any line

    return "In"  # Ball is in if it doesn't intersect any line

# Check the ball's in/out status
ball_status = check_ball_in_out(ball_center, lines)
print(f"Ball status: {ball_status}")

# --- Show Combined Result ---
plt.figure(figsize=(8, 6))
plt.imshow(lineimg)
plt.title('Court Lines and Ball Outline')
plt.axis('off')
plt.show()
