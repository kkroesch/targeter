import os

import cv2

from analyzer import (
    detect_and_draw_rings,
    detect_holes_by_color,
    evaluate_and_draw_scores,
    find_and_warp_target,
    read_target_number,
)

# --- Configuration ---
IMAGE_PATH = "target.jpeg"
TARGET_SIZE = 800


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
