# First Code, it worked but not just on pennies.
import os
import cv2
import numpy as np


def count_coins(image_path: str, out_name: str = "coins_annotated.png") -> int:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    original = img.copy()

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold: coins become white
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)

    # Distance transform for coin centers
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Tune this value if needed: 0.35–0.60
    _, sure_fg = cv2.threshold(dist_norm, 0.45, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    # Count segments (labels > 1)
    coin_labels = [label for label in np.unique(markers) if label > 1]
    coin_count = len(coin_labels)

    # Annotate
    annotated = original.copy()
    for label in coin_labels:
        mask = (markers == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:
            continue

        cv2.drawContours(annotated, [c], -1, (0, 255, 0), 2)
        M = cv2.moments(c)
        if M["m00"]:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                annotated,
                str(label - 1),
                (cx - 10, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

    cv2.putText(
        annotated,
        f"Count: {coin_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 255),
        3,
    )

    out_path = os.path.join(os.path.dirname(image_path), out_name)
    cv2.imwrite(out_path, annotated)
    return coin_count


if __name__ == "__main__":
    image_path = os.path.join("coins", "coins.png")
    count = count_coins(image_path, out_name="coins_annotated.png")
    print(f"Pennies counted: {count}")
    print("Saved annotated image to: coins/coins_annotated.png")


# 2 This code count the pennies correctly but do not highlight them.
import os
import cv2
import numpy as np


def is_penny(coin_bgr: np.ndarray, coin_mask: np.ndarray) -> bool:
    """
    Decide if a segmented coin region is a penny (copper/brown) using color stats.

    We use:
      - Lab 'a*' channel: red/green axis (copper tends to push higher a*)
      - HSV saturation: copper coins usually have higher saturation than silver coins

    coin_mask should be 0/255 with coin area = 255.
    """
    # Masked pixels only
    mask = (coin_mask > 0).astype(np.uint8) * 255

    # Lab color space
    lab = cv2.cvtColor(coin_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]  # a* channel (higher = more reddish)

    # HSV color space
    hsv = cv2.cvtColor(coin_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]  # saturation

    # Compute means within the coin
    a_mean = cv2.mean(a, mask=mask)[0]
    s_mean = cv2.mean(s, mask=mask)[0]

    # --- Thresholds to classify pennies ---
    # These are reasonable defaults; tweak if needed:
    # - If you miss pennies: lower a_thresh slightly (e.g., 136->132)
    # - If you incorrectly include silver coins: raise a_thresh (e.g., 136->140)
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

    # TUNE if coins touching merge/split:
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

    # labels: -1 boundaries, 1 background, >=2 regions
    for label in np.unique(markers):
        if label <= 1:
            continue

        region_mask = (markers == label).astype(np.uint8) * 255

        # Clean region to avoid jagged boundary artifacts
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

        # Crop ROI for color classification (faster + better stats)
        x, y, w, h = cv2.boundingRect(c)
        pad = 3
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

        coin_roi = original[y0:y1, x0:x1]
        mask_roi = region_mask[y0:y1, x0:x1]

        if is_penny(coin_roi, mask_roi):
            penny_count += 1

            # Draw ONLY pennies with a copper/brown outline (BGR)
            brown = (60, 110, 200)  # tweak if you want different brown
            cv2.drawContours(annotated, [c], -1, brown, 3)

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
                    brown,
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