# Exp-7-Record-HOUGH-TRANSFORM
# Aim
To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

# Learning Objective
Understand each stage of image processing
Learn how to build a complete computer vision pipeline
Practice writing code in guided sections
Important Instruction: 👉 Write code ONLY in places marked as # Your Code Here 👉 Do NOT modify any other part of the code

# Instructions
Fill ONLY in # Your Code Here sections
Do NOT change existing code
Run step-by-step
Verify outputs
# Software Used
Anaconda – Python 3.7
Jupyter Notebook / VS Code
OpenCV (cv2)
NumPy
Matplotlib
# Developed By
Name: Suvetha R
Register No: 2122252320113
# Algorithm & Explanation
# Step 1: Import Libraries
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
# Step 2: Read the Image
```
img = cv2.imread("chess.png")
```
# Step 3: Convert to Grayscale
```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
# Step 4: Display Images
```
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
```
# Step 5: Thresholding
```
threshold = int(np.percentile(gray, 90))
print("Threshold used:", threshold)
_, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6, 5))
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()
```
# Step 6: Region of Interest (ROI)
```
height, width = thresh.shape

# Trapezoid covering the road ahead
polygon = np.array([[
    (0, height),
    (width, height),
    (int(0.55 * width), int(0.6 * height)),
    (int(0.45 * width), int(0.6 * height))
]], dtype=np.int32)

mask = np.zeros_like(thresh)
cv2.fillPoly(mask, polygon, 255)
masked = cv2.bitwise_and(thresh, mask)

plt.figure(figsize=(6, 5))
plt.imshow(masked, cmap="gray")
plt.title("ROI Masked Image")
plt.axis("off")
plt.show()
```
# Step 7: Edge Detection (Canny)
```
edges = cv2.Canny(masked, 50, 150)

plt.figure(figsize=(6, 5))
plt.imshow(edges, cmap="gray")
plt.title("Edge Detected Image")
plt.axis("off")
plt.show()
```
# Step 8: Gaussian Blur
```
blur = cv2.GaussianBlur(edges, (5, 5), 0)

plt.figure(figsize=(6, 5))
plt.imshow(blur, cmap="gray")
plt.title("Smoothed Image")
plt.axis("off")
plt.show()
```
# Step 9: Hough Transform
```
lines = cv2.HoughLinesP(blur, 1, np.pi / 180, threshold=50,
                        minLineLength=50, maxLineGap=150)

line_img = np.zeros_like(img)
if lines is not None:
    lines = lines.reshape(-1, 4)
    for x1, y1, x2, y2 in lines:
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines")
plt.axis("off")
plt.show()
```
# Step 10: Lane Detection Logic
```
left_fit = []
right_fit = []

if lines is not None:
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        # Image y-axis points down: left lane has negative slope, right lane positive
        if slope < -0.3:
            left_fit.append((slope, intercept))
        elif slope > 0.3:
            right_fit.append((slope, intercept))

def make_points(y_bottom, y_top, fit):
    slope, intercept = fit
    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return (x_bottom, y_bottom, x_top, y_top)

h = img.shape[0]
y_bottom = h
y_top = int(h * 0.6)

lane_img = np.zeros_like(img)

if len(left_fit) > 0:
    x1, y1, x2, y2 = make_points(y_bottom, y_top, np.mean(left_fit, axis=0))
    cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 8)

if len(right_fit) > 0:
    x1, y1, x2, y2 = make_points(y_bottom, y_top, np.mean(right_fit, axis=0))
    cv2.line(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 8)

final = cv2.addWeighted(img, 0.8, lane_img, 1, 0)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title("Final Lane Detection Output")
plt.axis("off")
plt.show()
```
# Expected Output
# Original image
<img width="311" height="280" alt="image" src="https://github.com/user-attachments/assets/0a7da60b-2af6-4081-a924-75bf74fcdf4b" />

# Grayscale image
<img width="322" height="282" alt="image" src="https://github.com/user-attachments/assets/5bd90b54-7aed-4ad7-865f-5b1cb3e80962" />

# Thresholded image
<img width="602" height="521" alt="image" src="https://github.com/user-attachments/assets/c4374378-d439-4dcf-8444-6dc6770ecef9" />

# ROI masked image
<img width="615" height="517" alt="image" src="https://github.com/user-attachments/assets/d3b31a03-130d-4e7a-a423-e6fe488279b0" />

# Edge detected image
<img width="640" height="507" alt="image" src="https://github.com/user-attachments/assets/2ee29380-af5a-4cc2-ac7a-3d28d832c064" />

# Smoothed image
<img width="626" height="557" alt="image" src="https://github.com/user-attachments/assets/98084de4-5e89-41cb-8835-8ba8da649b78" />

# Detected lines
<img width="605" height="500" alt="image" src="https://github.com/user-attachments/assets/af83f4f3-8e20-4836-988e-fde617a922ce" />

# Final lane detection output
<img width="735" height="615" alt="image" src="https://github.com/user-attachments/assets/cc92edde-62f5-464b-bd8e-2167046ee27c" />

# Result
Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

