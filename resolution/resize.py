import cv2 as cv
import os
import glob

TARGET_W = 640

def resize_keep_aspect(img, target_w):
    h, w = img.shape[:2]
    ratio = w / target_w               # horizontal reduction ratio
    target_h = int(round(h / ratio))   # scale vertical accordingly
    resized = cv.resize(img, (target_w, target_h), interpolation=cv.INTER_AREA)
    return resized, target_h

def main():
    # Make sure we run from the repo root (CIS450-demo)
    photos_dir = "photos"
    out_dir = "resolution"
    os.makedirs(out_dir, exist_ok=True)

    # Grab common image types
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(photos_dir, p)))

    # ✅ REMOVE DUPLICATES (THIS IS THE ONLY CHANGE)
    files = sorted(set(files))

    if not files:
        print(f"No images found in {photos_dir}/")
        return

    print(f"Found {len(files)} images")

    for path in files:
        filename = os.path.basename(path)
        name_no_ext, _ = os.path.splitext(filename)

        img = cv.imread(path)
        if img is None:
            print(f"SKIP (could not read): {path}")
            continue

        resized, target_h = resize_keep_aspect(img, TARGET_W)

        out_name = f"{name_no_ext}-640x{target_h}.png"
        out_path = os.path.join(out_dir, out_name)

        ok = cv.imwrite(out_path, resized)
        if ok:
            print(f"Saved: {out_path}")
        else:
            print(f"FAILED to save: {out_path}")

if __name__ == "__main__":
    main()

