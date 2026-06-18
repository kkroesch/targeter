import cv2
import numpy as np
import pytesseract

# --- Konfiguration ---
TARGET_SIZE = 800
HOLE_COLOR_LOWER = np.array([5, 120, 120])  # Orange
HOLE_COLOR_UPPER = np.array([25, 255, 255])

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
    pts_dst = np.float32([[0, 0], [TARGET_SIZE - 1, 0], [TARGET_SIZE - 1, TARGET_SIZE - 1], [0, TARGET_SIZE - 1]])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(image, matrix, (TARGET_SIZE, TARGET_SIZE))

def read_target_number(warped_img):
    roi = warped_img[10:100, 10:300]
    roi_scaled = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_scaled, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    _, bw = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
    return pytesseract.image_to_string(bw, config=config).strip()

def detect_and_draw_rings(warped_img):
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, mask_black = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    raw_radii = [cv2.minEnclosingCircle(cnt)[1] for cnt in contours_edges if len(cnt) >= 5 and abs(cv2.minEnclosingCircle(cnt)[0][0] - cx) < 10 and abs(cv2.minEnclosingCircle(cnt)[0][1] - cy) < 10 and cv2.minEnclosingCircle(cnt)[1] > 15]
    
    raw_radii.sort()
    unique_radii = []
    for r in raw_radii:
        if not unique_radii or r - unique_radii[-1] > 12:
            unique_radii.append(r)
            
    output_img = warped_img.copy()
    cv2.circle(output_img, (cx, cy), 4, (0, 0, 255), -1)
    for r in unique_radii:
        cv2.circle(output_img, (cx, cy), int(r), (255, 0, 0), 2)
    return output_img, (cx, cy), unique_radii

def detect_holes_by_color(warped_img):
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HOLE_COLOR_LOWER, HOLE_COLOR_UPPER)
    kernel = np.ones((3,3), np.uint8)
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

def evaluate_and_draw_scores(image, holes, center, radii):
    if not holes or not radii or center is None:
        return image, []
    
    cx, cy = center
    scores = []
    HOLE_RADIUS = 12 
    
    for hx, hy in holes:
        dist = np.sqrt((hx - cx)**2 + (hy - cy)**2)
        score = 0
        
        for i, r in enumerate(radii):
            # ISSF-Regel: Ankratzen der Linie reicht für höheren Wert
            if (dist - HOLE_RADIUS) <= r:
                # Bei 11 Ringen (inkl. Mouche) wird das Maximum auf 10 gedeckelt
                score = min(10, len(radii) - i)
                break
                
        scores.append({'coord': (hx, hy), 'dist': dist, 'score': score})
        
    scores.sort(key=lambda x: x['dist'])
    
    output_img = image.copy()
    total = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Overlay-Box für bessere Lesbarkeit der Tabelle
    overlay = output_img.copy()
    box_x, box_y = 690, 20
    box_w, box_h = 100, 50 + len(scores) * 30
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.8, output_img, 0.2, 0, output_img)

    table_x, table_y, line_height = 700, 50, 30 
    
    for i, item in enumerate(scores):
        hx, hy = item['coord']
        pts = item['score']
        total += pts
        
        cv2.circle(output_img, (hx, hy), 12, (0, 0, 255), 2)
        cv2.circle(output_img, (hx, hy), 2, (0, 0, 255), -1)
        cv2.putText(output_img, str(i + 1), (hx + 15, hy - 15), font, 0.6, (0, 0, 255), 2)
        
        # Formatierte Zeile (z.B. "1: 10")
        row_text = f"{i + 1}: {pts}" 
        cv2.putText(output_img, row_text, (table_x, table_y + i * line_height), font, 0.6, (0, 0, 0), 2)
    
    # Trennlinie und Summe
    line_y = table_y + len(scores) * line_height - 15
    cv2.line(output_img, (box_x + 10, line_y), (box_x + box_w - 10, line_y), (150, 150, 150), 1)
    
    total_y = table_y + len(scores) * line_height + 10
    cv2.putText(output_img, f"Sum: {total}", (table_x, total_y), font, 0.65, (0, 0, 255), 2)
    
    return output_img, scores