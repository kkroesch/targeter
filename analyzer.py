import cv2
import numpy as np
import pytesseract

# --- Konfiguration & Konstanten ---
TARGET_SIZE = 800
HOLE_COLOR_LOWER = np.array([5, 120, 120])  # Orange
HOLE_COLOR_UPPER = np.array([25, 255, 255])

# --- UI & Layout Konstanten ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GRAY = (150, 150, 150)
COLOR_BLUE = (255, 0, 0)

UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_FONT_SCALE_ROW = 0.6
UI_FONT_SCALE_SUM = 0.65
UI_THICKNESS = 2

TABLE_BOX_X = 690
TABLE_BOX_Y = 20
TABLE_BOX_W = 100
TABLE_TEXT_X = 700
TABLE_TEXT_START_Y = 50
TABLE_LINE_HEIGHT = 30
TABLE_OVERLAY_ALPHA = 0.8

HOLE_MARKER_RADIUS = 12
HOLE_CENTER_RADIUS = 2
MPI_SQUARE_SIZE = 8


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_and_warp_target(image):
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
        raise ValueError("Zielscheibe nicht gefunden.")

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
    return cv2.warpPerspective(image, matrix, (TARGET_SIZE, TARGET_SIZE))


def read_target_number(warped_img):
    roi = warped_img[10:100, 10:300]
    roi_scaled = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    _, bw = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    return pytesseract.image_to_string(bw, config=config).strip()


def detect_and_draw_rings(warped_img):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, mask_black = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    contours_black, _ = cv2.findContours(
        mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours_black:
        return warped_img, None, []
    spiegel_cnt = max(contours_black, key=cv2.contourArea)
    M = cv2.moments(spiegel_cnt)
    if M["m00"] == 0:
        return warped_img, None, []
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    contours_edges, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw_radii = [
        cv2.minEnclosingCircle(cnt)[1]
        for cnt in contours_edges
        if len(cnt) >= 5
        and abs(cv2.minEnclosingCircle(cnt)[0][0] - cx) < 10
        and abs(cv2.minEnclosingCircle(cnt)[0][1] - cy) < 10
        and cv2.minEnclosingCircle(cnt)[1] > 15
    ]

    raw_radii.sort()
    unique_radii = []
    for r in raw_radii:
        if not unique_radii or r - unique_radii[-1] > 12:
            unique_radii.append(r)

    output_img = warped_img.copy()
    cv2.circle(output_img, (cx, cy), 4, COLOR_RED, -1)
    for r in unique_radii:
        cv2.circle(output_img, (cx, cy), int(r), COLOR_BLUE, UI_THICKNESS)
    return output_img, (cx, cy), unique_radii


def detect_holes_by_color(warped_img):
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HOLE_COLOR_LOWER, HOLE_COLOR_UPPER)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    SINGLE_HOLE_AREA_EXPECTED = 250
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            shots_in_cluster = max(1, int(round(area / SINGLE_HOLE_AREA_EXPECTED)))
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                for _ in range(shots_in_cluster):
                    holes.append((cx, cy))
    return holes


def calculate_mean_point_of_impact(holes):
    """Berechnet den Schwerpunkt aller Treffer."""
    if not holes:
        return None
    mean_x = int(sum(h[0] for h in holes) / len(holes))
    mean_y = int(sum(h[1] for h in holes) / len(holes))
    return (mean_x, mean_y)


def draw_score_table(image, scores, total_score):
    """Zeichnet die Punktetabelle mit transparentem Hintergrund oben rechts."""
    output_img = image.copy()

    # Hintergrund-Box berechnen und zeichnen
    box_h = 50 + len(scores) * TABLE_LINE_HEIGHT
    overlay = output_img.copy()
    cv2.rectangle(
        overlay,
        (TABLE_BOX_X, TABLE_BOX_Y),
        (TABLE_BOX_X + TABLE_BOX_W, TABLE_BOX_Y + box_h),
        COLOR_WHITE,
        -1,
    )
    cv2.addWeighted(
        overlay, TABLE_OVERLAY_ALPHA, output_img, 1 - TABLE_OVERLAY_ALPHA, 0, output_img
    )

    # Zeilen für jeden Schuss
    for i, item in enumerate(scores):
        pts = item["score"]
        row_text = f"{i + 1}: {pts}"
        y_pos = TABLE_TEXT_START_Y + i * TABLE_LINE_HEIGHT
        cv2.putText(
            output_img,
            row_text,
            (TABLE_TEXT_X, y_pos),
            UI_FONT,
            UI_FONT_SCALE_ROW,
            COLOR_BLACK,
            UI_THICKNESS,
        )

    # Trennlinie
    line_y = TABLE_TEXT_START_Y + len(scores) * TABLE_LINE_HEIGHT - 15
    cv2.line(
        output_img,
        (TABLE_BOX_X + 10, line_y),
        (TABLE_BOX_X + TABLE_BOX_W - 10, line_y),
        COLOR_GRAY,
        1,
    )

    # Summe
    total_y = TABLE_TEXT_START_Y + len(scores) * TABLE_LINE_HEIGHT + 10
    cv2.putText(
        output_img,
        f"Sum: {total_score}",
        (TABLE_TEXT_X, total_y),
        UI_FONT,
        UI_FONT_SCALE_SUM,
        COLOR_RED,
        UI_THICKNESS,
    )

    return output_img


def evaluate_and_draw_scores(image, holes, center, radii):
    if not holes or not radii or center is None:
        return image, []

    cx, cy = center
    scores = []

    # 1. Distanzen und Punkte berechnen
    for hx, hy in holes:
        dist = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
        score = 0

        for i, r in enumerate(radii):
            if (dist - HOLE_MARKER_RADIUS) <= r:
                score = min(10, len(radii) - i)
                break

        scores.append({"coord": (hx, hy), "dist": dist, "score": score})

    scores.sort(key=lambda x: x["dist"])

    # 2. Treffer ins Bild zeichnen
    output_img = image.copy()
    total_score = 0

    for i, item in enumerate(scores):
        hx, hy = item["coord"]
        total_score += item["score"]

        # Loch-Markierung
        cv2.circle(output_img, (hx, hy), HOLE_MARKER_RADIUS, COLOR_RED, UI_THICKNESS)
        cv2.circle(output_img, (hx, hy), HOLE_CENTER_RADIUS, COLOR_RED, -1)
        # Schussnummer neben dem Loch
        cv2.putText(
            output_img,
            str(i + 1),
            (hx + 15, hy - 15),
            UI_FONT,
            UI_FONT_SCALE_ROW,
            COLOR_RED,
            UI_THICKNESS,
        )

    # 3. Trefferschwerpunkt (MPI) einzeichnen
    mpi = calculate_mean_point_of_impact(holes)
    if mpi:
        mx, my = mpi
        cv2.rectangle(
            output_img,
            (mx - MPI_SQUARE_SIZE, my - MPI_SQUARE_SIZE),
            (mx + MPI_SQUARE_SIZE, my + MPI_SQUARE_SIZE),
            COLOR_RED,
            UI_THICKNESS,
        )

    # 4. Tabelle zeichnen
    output_img = draw_score_table(output_img, scores, total_score)

    return output_img, scores
