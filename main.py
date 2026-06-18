import os

import cv2
import numpy as np
import pytesseract

# --- Konfiguration ---
IMAGE_PATH = "target.jpeg"
TARGET_SIZE = 800

# HSV-Farbbereich für den Hintergrundkarton
# Bitte später an deine reale Pappe anpassen!
# HSV-Farbbereich für ORANGE
HOLE_COLOR_LOWER = np.array([5, 120, 120])
HOLE_COLOR_UPPER = np.array([25, 255, 255])
# ---------------------


def order_points(pts):
    """Sortiert 4 Koordinaten: Oben-Links, Oben-Rechts, Unten-Rechts, Unten-Links"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_and_warp_target(image):
    """Sucht die Pappe im Bild und entzerrt sie auf 800x800 Pixel."""
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
    """Detektiert das echte Zentrum und zeichnet die Ringe ein."""
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, mask_black = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    contours_black, _ = cv2.findContours(
        mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours_black:
        print("Warnung: Schwarzer Spiegel nicht gefunden.")
        return warped_img, None, []  # Immer 3 Werte zurückgeben!

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

    # Zeichnen
    output_img = warped_img.copy()
    cv2.circle(output_img, (cx, cy), 4, (0, 0, 255), -1)
    for r in unique_radii:
        cv2.circle(output_img, (cx, cy), int(r), (255, 0, 0), 2)

    return output_img, (cx, cy), unique_radii


def read_target_number(warped_img):
    """Schneidet den Bereich oben links aus und liest die Nummer via OCR."""
    # ROI (Region of Interest) oben links ausschneiden
    # Bei 800x800 liegt die Nummer ca. zwischen y:20-120 und x:20-300
    roi = warped_img[20:120, 20:300]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Otsu-Threshold für maximalen Schwarz-Weiß-Kontrast der Schrift
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract konfigurieren: Nur Ziffern (Whitelist) und PSM 7 (Einzelne Textzeile)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(bw, config=config)

    # Whitespaces/Zeilenumbrüche bereinigen
    clean_text = text.strip()
    return clean_text


def detect_holes_by_color(warped_img):
    """Sucht nach dem farbigen Hintergrundkarton und löst Cluster auf."""
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HOLE_COLOR_LOWER, HOLE_COLOR_UPPER)

    # Morphologisches Öffnen: Entfernt winziges Rauschen (Papierfasern), behält die Löcher
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = []

    # --- Konfiguration für Cluster ---
    # Bei 800x800 Pixeln hat ein einzelnes 4.5mm/5.6mm Loch meist grob diese Fläche:
    SINGLE_HOLE_AREA_EXPECTED = 250
    MIN_AREA = 50  # Alles darunter ist nur Staub

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            # Berechne, wie viele Schüsse in diesem Fleck stecken (aufrunden)
            shots_in_cluster = max(1, int(round(area / SINGLE_HOLE_AREA_EXPECTED)))

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Trage für jeden geschätzten Schuss im Cluster einen Treffer ein
                for _ in range(shots_in_cluster):
                    holes.append((cx, cy))

    if holes:
        print(
            f"-> {len(holes)} Schussloch/-löcher über Farbe detektiert (inkl. Cluster-Schätzung)."
        )

    return holes


def evaluate_and_draw_scores(image, holes, center, radii):
    # Wenn keine Löcher da sind, brich sauber ab
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

        # --- HIER IST DIE ANPASSUNG ---
        # 1. Roter Kreis um das Loch (Radius 12, Farbe Rot, Liniendicke 2)
        cv2.circle(output_img, (hx, hy), 12, (0, 0, 255), 2)

        # 2. Optional: Ein winziger roter Punkt exakt im Zentrum
        cv2.circle(output_img, (hx, hy), 2, (0, 0, 255), -1)

        # Nummerierung neben dem Loch (jetzt auch in Rot für bessere Lesbarkeit)
        cv2.putText(
            output_img, str(i + 1), (hx + 15, hy - 15), font, 0.6, (0, 0, 255), 2
        )

        # Tabelle oben rechts (bleibt schwarz)
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
    print(f"Lade {IMAGE_PATH}...")
    original = cv2.imread(IMAGE_PATH)

    if original is None:
        print(f"Fehler: Konnte {IMAGE_PATH} nicht laden.")
        return

    try:
        # 1. Entzerren
        warped = find_and_warp_target(original)
        print("-> Bild erfolgreich entzerrt.")

        # 2. Nummer OCR
        number = read_target_number(warped)
        if number:
            print(f"-> Nummer erkannt: {number}")
            filename = f"target_{number}.png"
            csv_filename = f"{number}.csv"
        else:
            print("-> Keine Nummer erkannt.")
            filename = "target_unknown.png"
            csv_filename = "unknown.csv"

        # 3. Ringe detektieren (WICHTIG: In 3 Variablen entpacken)
        img_with_rings, center, radii = detect_and_draw_rings(warped)

        # 4. Löcher suchen (WICHTIG: Über das saubere "warped" Bild suchen, nicht über das Tupel)
        holes = detect_holes_by_color(warped)

        # 5. Auswerten
        final_img, scores_data = evaluate_and_draw_scores(
            img_with_rings, holes, center, radii
        )

        # 6. Bild speichern
        cv2.imwrite(filename, final_img)
        print(f"-> Fertiges Bild gespeichert unter: {filename}")

        # 7. CSV speichern
        with open(csv_filename, "w", encoding="utf-8") as f:
            for i, item in enumerate(scores_data):
                f.write(f"{i + 1};{item['score']}\n")
        print(f"-> CSV exportiert unter: {csv_filename} (Treffer: {len(scores_data)})")

        cv2.imshow("Finale Analyse", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        # Erweitertes Error-Printing hilft enorm bei der Fehlersuche
        import traceback

        traceback.print_exc()
        print(f"Fehler in der Pipeline: {e}")


if __name__ == "__main__":
    main()
