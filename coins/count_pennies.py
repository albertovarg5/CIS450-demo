import os
import cv2
import numpy as np


def is_penny(coin_bgr: np.ndarray, coin_mask: np.ndarray) -> bool:
    """
    Decide if a segmented coin region is a penny (copper/brown) using color stats.

    Uses:
      - Lab 'a*' channel (reddishness)
      - HSV saturation (copper tends to be more saturated than silver coins)

    coin_mask should be 0/255 with coin area = 255.
    """
    mask = (coin_mask > 0).astype(np.uint8) * 255

    lab = cv2.cvtColor(coin_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]  # a* channel

    hsv = cv2.cvtColor(coin_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]  # saturation

    a_mean = cv2.mean(a, mask=mask)[0]
    s_mean = cv2.mean(s, mask=mask)[0]

    # Tune if needed:
    # - If you miss pennies: lower slightly (e.g., 136 -> 132)
    # - If you include silver coins: raise slightly (e.g., 136 -> 140)
    a_thresh = 136
    s_thresh = 55

    return (a_mean >= a_thresh) and (s_mean >= s_thresh)


def count_pennies(image_path: str, out_name: str = "coins_annotated.png") -> int:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original = img.copy()

    # ---- SEGMENT COINS (WATERSHED) ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # coins -> white
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Tune if coins touching merge/split:
    # - merging: lower (0.45 -> 0.40)
    # - splitting: raise (0.45 -> 0.50)
    _, sure_fg = cv2.threshold(dist_norm, 0.45, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    # ---- CLASSIFY + ANNOTATE ONLY PENNIES ----
    annotated = original.copy()
    penny_count = 0
    label_num = 1

    green = (0, 255, 0)  # BGR pure green for highlighting pennies

    for label in np.unique(markers):
        if label <= 1:
            continue

        region_mask = (markers == label).astype(np.uint8) * 255

        # Smooth the region mask a bit to avoid jagged boundaries
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)

        # Filter tiny junk
        if area < 1500:
            continue

        # Crop ROI for color classification
        x, y, w, h = cv2.boundingRect(c)
        pad = 3
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

        coin_roi = original[y0:y1, x0:x1]
        mask_roi = region_mask[y0:y1, x0:x1]

        if is_penny(coin_roi, mask_roi):
            penny_count += 1

            # Draw ONLY pennies in green
            cv2.drawContours(annotated, [c], -1, green, 3)

            # Label penny number at centroid
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    annotated,
                    str(label_num),
                    (cx - 10, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    green,
                    2,
                )
                label_num += 1

    # Title text
    cv2.putText(
        annotated,
        f"Pennies: {penny_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
    )

    out_path = os.path.join(os.path.dirname(image_path), out_name)
    cv2.imwrite(out_path, annotated)
    return penny_count


if __name__ == "__main__":
    image_path = os.path.join("coins", "coins.png")
    pennies = count_pennies(image_path, out_name="coins_annotated.png")
    print(f"Pennies counted: {pennies}")
    print("Saved annotated image to: coins/coins_annotated.png")