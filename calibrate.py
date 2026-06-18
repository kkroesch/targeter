import os
from datetime import datetime

import cv2
import numpy as np

# Globale Listen und Variablen
points = []
image_base = None


def click_event(event, x, y, flags, param):
    """Mouse-Callback für Punktauswahl und Fadenkreuz."""
    global points, image_base

    if len(points) >= 5:
        return

    # Fadenkreuz bei Mausbewegung zeichnen
    if event == cv2.EVENT_MOUSEMOVE:
        temp_img = image_base.copy()

        # Fadenkreuz (Kreuz + mittiger Kreis)
        cv2.line(temp_img, (x - 20, y), (x + 20, y), (0, 0, 255), 1)
        cv2.line(temp_img, (x, y - 20), (x, y + 20), (0, 0, 255), 1)
        cv2.circle(temp_img, (x, y), 6, (0, 0, 255), 1)

        cv2.imshow("Original", temp_img)

    # Punkt bei Klick speichern
    elif event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Fest in das Basisbild einzeichnen (grüner Punkt)
        cv2.circle(image_base, (x, y), 5, (0, 255, 0), -1)

        if len(points) == 5:
            print("\n5 Punkte markiert! Drücke eine beliebige Taste im Bildfenster.")
            cv2.imshow("Original", image_base)


# 1. Bild laden
image_path = "target.jpeg"
image_base = cv2.imread(image_path)

if image_base is None:
    print(f"Fehler: Konnte {image_path} nicht laden.")
    exit()

# 2. Interaktive Auswahl
cv2.namedWindow("Original")
cv2.setMouseCallback("Original", click_event)

print(
    "Klicke 5 Punkte: Oben-Links -> Oben-Rechts -> Unten-Rechts -> Unten-Links -> Mitte."
)
print("Drücke danach eine beliebige Taste, um fortzufahren.")

cv2.imshow("Original", image_base)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Verarbeitung und Speichern (wenn 5 Punkte vorliegen)
if len(points) == 5:
    # Nur die ersten 4 Punkte für die Entzerrung nutzen
    pts_src = np.float32(points[:4])

    size = 800
    pts_dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])

    # Transformation
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image_base, matrix, (size, size))

    # Ordnerstruktur anlegen: target_YYYY-MM-DD
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = f"target_{today}"
    os.makedirs(out_dir, exist_ok=True)

    # Bild speichern
    img_output_path = os.path.join(out_dir, "target_ref.png")
    cv2.imwrite(img_output_path, warped)

    # Koordinaten speichern
    coords_output_path = os.path.join(out_dir, "coordinates.txt")
    with open(coords_output_path, "w") as f:
        for i, pt in enumerate(points):
            label = f"Ecke_{i + 1}" if i < 4 else "Mitte"
            f.write(f"{label}: {pt[0]}, {pt[1]}\n")

    print(f"Erfolgreich gespeichert unter '{out_dir}/'.")

    cv2.imshow("Zielscheibe Ausschnitt", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Fehler: Es wurden {len(points)} Punkte markiert. Genau 5 sind nötig.")
