import os
from datetime import datetime

import cv2
import numpy as np

# Global lists and variables
points = []
image_base = None


def click_event(event, x, y, flags, param):
    """Mouse callback for point selection and crosshair."""
    global points, image_base

    if len(points) >= 5:
        return

    # Draw crosshair on mouse movement
    if event == cv2.EVENT_MOUSEMOVE:
        temp_img = image_base.copy()

        # Crosshair (cross + center circle)
        cv2.line(temp_img, (x - 20, y), (x + 20, y), (0, 0, 255), 1)
        cv2.line(temp_img, (x, y - 20), (x, y + 20), (0, 0, 255), 1)
        cv2.circle(temp_img, (x, y), 6, (0, 0, 255), 1)

        cv2.imshow("Original", temp_img)

    # Save point on click
    elif event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Draw permanently on the base image (green point)
        cv2.circle(image_base, (x, y), 5, (0, 255, 0), -1)

        if len(points) == 5:
            print("\n5 points marked! Press any key in the image window.")
            cv2.imshow("Original", image_base)


# 1. Load image
image_path = "target.jpeg"
image_base = cv2.imread(image_path)

if image_base is None:
    print(f"Error: Could not load {image_path}.")
    exit()

# 2. Interactive selection
cv2.namedWindow("Original")
cv2.setMouseCallback("Original", click_event)

print("Click 5 points: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left -> Center.")
print("Then press any key to continue.")

cv2.imshow("Original", image_base)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Process and save (if 5 points are present)
if len(points) == 5:
    # Use only the first 4 points for warping
    pts_src = np.float32(points[:4])

    size = 800
    pts_dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])

    # Transformation
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image_base, matrix, (size, size))

    # Create directory structure: target_YYYY-MM-DD
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = f"target_{today}"
    os.makedirs(out_dir, exist_ok=True)

    # Save image
    img_output_path = os.path.join(out_dir, "target_ref.png")
    cv2.imwrite(img_output_path, warped)

    # Save coordinates
    coords_output_path = os.path.join(out_dir, "coordinates.txt")
    with open(coords_output_path, "w") as f:
        for i, pt in enumerate(points):
            label = f"Corner_{i + 1}" if i < 4 else "Center"
            f.write(f"{label}: {pt[0]}, {pt[1]}\n")

    print(f"Successfully saved under '{out_dir}/'.")

    cv2.imshow("Target Crop", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Error: {len(points)} points were marked. Exactly 5 are required.")
