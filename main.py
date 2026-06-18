import os

import cv2
import numpy as np
import pytesseract

# --- Configuration ---
IMAGE_PATH = "target.jpeg"
TARGET_SIZE = 800

# HSV color range for the background cardboard
# Please adjust to your actual cardboard later!
# HSV color range for ORANGE
HOLE_COLOR_LOWER = np.array([5, 120, 120])
HOLE_COLOR_UPPER = np.array([25, 255, 255])
# ---------------------


def order_points(pts):
    """Sorts 4 coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_and_warp_target(image):
    """Finds the cardboard in the image and warps it to 800x800 pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    target_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            target_contour = approx
            break

    if target_contour is None:
        raise ValueError("Konnte die 4 Ecken der Zielscheibe nicht finden.")

    pts_src = order_points(target_contour.reshape(4, 2))
    pts_dst = np.float32(
        [
            [0, 0],
            [TARGET_SIZE - 1, 0],
            [TARGET_SIZE - 1, TARGET_SIZE - 1],
            [0, TARGET_SIZE - 1],
        ]
    )

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, matrix, (TARGET_SIZE, TARGET_SIZE))

    return warped


def detect_and_draw_rings(warped_img):
    """Detects the actual center and draws the rings."""
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, mask_black = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    contours_black, _ = cv2.findContours(
        mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours_black:
        print("Warnung: Schwarzer Spiegel nicht gefunden.")
        return warped_img, None, []  # Always return 3 values!

    spiegel_cnt = max(contours_black, key=cv2.contourArea)
    M = cv2.moments(spiegel_cnt)

    if M["m00"] == 0:
        return warped_img, None, []

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    contours_edges, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw_radii = []
    for cnt in contours_edges:
        if len(cnt) >= 5:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if abs(x - cx) < 10 and abs(y - cy) < 10 and radius > 15:
                raw_radii.append(radius)

    raw_radii.sort()
    unique_radii = []
    for r in raw_radii:
        if not unique_radii or r - unique_radii[-1] > 12:
            unique_radii.append(r)

    # Drawing
    output_img = warped_img.copy()
    cv2.circle(output_img, (cx, cy), 4, (0, 0, 255), -1)
    for r in unique_radii:
        cv2.circle(output_img, (cx, cy), int(r), (255, 0, 0), 2)

    return output_img, (cx, cy), unique_radii


def read_target_number(warped_img):
    """Crops the area, connects stamp dots, and reads the OCR."""
    # 1. Select a slightly larger region (y: 10-100, x: 10-300)
    roi = warped_img[10:100, 10:300]

    # 2. Scale (3x larger)
    roi_scaled = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2GRAY)

    # 3. Morphological closing: connects the dots of the stamp text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 4. Otsu thresholding for hard contrast
    _, bw = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- DEBUG WINDOW ---
    # cv2.imshow("DEBUG OCR - press a key to continue", bw)
    # cv2.waitKey(0)
    # ---------------------

    # 5. Tesseract: Single line (PSM 7) and digits only
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(bw, config=config)

    return text.strip()


def detect_holes_by_color(warped_img):
    """Searches for the colored background cardboard and resolves clusters."""
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HOLE_COLOR_LOWER, HOLE_COLOR_UPPER)

    # Morphological opening: removes tiny noise (paper fibers), keeps the holes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = []

    # --- Cluster configuration ---
    # At 800x800 pixels, a single 4.5mm/5.6mm hole typically has roughly this area:
    SINGLE_HOLE_AREA_EXPECTED = 250
    MIN_AREA = 50  # Everything below this is just dust

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            # Calculate how many shots are in this blob (round up)
            shots_in_cluster = max(1, int(round(area / SINGLE_HOLE_AREA_EXPECTED)))

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Record a hit for each estimated shot in the cluster
                for _ in range(shots_in_cluster):
                    holes.append((cx, cy))

    if holes:
        print(
            f"-> {len(holes)} bullet hole(s) detected via color (incl. cluster estimation)."
        )

    return holes


def evaluate_and_draw_scores(image, holes, center, radii):
    # If no holes exist, cleanly exit
    if not holes or not radii or center is None:
        return image, []

    cx, cy = center
    scores = []

    for hx, hy in holes:
        dist = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
        score = 0
        for i, r in enumerate(radii):
            if dist <= r:
                score = 10 - i
                break
        scores.append({"coord": (hx, hy), "dist": dist, "score": score})

    scores.sort(key=lambda x: x["dist"])

    output_img = image.copy()
    total = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    table_x, table_y, line_height = 730, 65, 43

    for i, item in enumerate(scores):
        hx, hy = item["coord"]
        pts = item["score"]
        total += pts

        # --- HERE IS THE ADJUSTMENT ---
        # 1. Red circle around the hole (radius 12, color red, line thickness 2)
        cv2.circle(output_img, (hx, hy), 12, (0, 0, 255), 2)

        # 2. Optional: a tiny red dot exactly at the center
        cv2.circle(output_img, (hx, hy), 2, (0, 0, 255), -1)

        # Numbering next to the hole (now also in red for better readability)
        cv2.putText(
            output_img, str(i + 1), (hx + 15, hy - 15), font, 0.6, (0, 0, 255), 2
        )

        # Table at top right (stays black)
        cv2.putText(
            output_img,
            str(pts),
            (table_x, table_y + i * line_height),
            font,
            0.9,
            (0, 0, 0),
            2,
        )

    total_y = table_y + len(scores) * line_height + 20
    cv2.putText(
        output_img, f"{total}", (table_x - 20, total_y), font, 0.9, (0, 0, 255), 2
    )

    return output_img, scores


def main():
    print(f"Loading {IMAGE_PATH}...")
    original = cv2.imread(IMAGE_PATH)

    if original is None:
        print(f"Error: Could not load {IMAGE_PATH}.")
        return

    try:
        # 1. Warp
        warped = find_and_warp_target(original)
        print("-> Image successfully warped.")

        # 2. Number OCR
        number = read_target_number(warped)
        if number:
            print(f"-> Number detected: {number}")
            filename = f"target_{number}.png"
            csv_filename = f"{number}.csv"
        else:
            print("-> No number detected.")
            filename = "target_unknown.png"
            csv_filename = "unknown.csv"

        # 3. Detect rings (IMPORTANT: Unpack into 3 variables)
        img_with_rings, center, radii = detect_and_draw_rings(warped)

        # 4. Search holes (IMPORTANT: Search over the clean "warped" image, not over the tuple)
        holes = detect_holes_by_color(warped)

        # 5. Evaluate
        final_img, scores_data = evaluate_and_draw_scores(
            img_with_rings, holes, center, radii
        )

        # 6. Save image
        cv2.imwrite(filename, final_img)
        print(f"-> Finished image saved as: {filename}")

        # 7. Export CSV
        with open(csv_filename, "w", encoding="utf-8") as f:
            for i, item in enumerate(scores_data):
                f.write(f"{i + 1};{item['score']}\n")
        print(f"-> CSV exported as: {csv_filename} (Hits: {len(scores_data)})")

        cv2.imshow("Finale Analyse", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        # Extended error printing helps immensely with debugging
        import traceback

        traceback.print_exc()
        print(f"Error in pipeline: {e}")


if __name__ == "__main__":
    main()
