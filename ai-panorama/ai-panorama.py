import cv2 as cv
from pathlib import Path

def collect_image_files(folder: Path):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(folder.glob(ext))
    return sorted(files)

def load_images(files, scale=0.6):
    imgs = []
    good_files = []
    for f in files:
        img = cv.imread(str(f))
        if img is None:
            continue
        if scale != 1.0:
            img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        imgs.append(img)
        good_files.append(f)
    return imgs, good_files

def stitch_panorama(imgs, confidence=0.7):
    stitcher = cv.Stitcher_create(cv.Stitcher_PANORAMA)
    stitcher.setPanoConfidenceThresh(confidence)
    status, pano = stitcher.stitch(imgs)
    return status, pano

def try_sets(files, scale=0.6):
    """
    Try:
    1) all images
    2) drop each single image (sometimes 1 bad photo breaks the chain)
    Returns best pano.
    """
    best = None
    best_used = []
    best_status = None

    # helper to try a file list
    def attempt(use_files):
        imgs, used_files = load_images(use_files, scale=scale)
        if len(imgs) < 2:
            return None, used_files, None
        for conf in (1.0, 0.8, 0.6, 0.4):
            status, pano = stitch_panorama(imgs, confidence=conf)
            if status == cv.Stitcher_OK and pano is not None:
                return pano, used_files, status
        return None, used_files, status

    # 1) try all
    pano, used, st = attempt(files)
    if pano is not None:
        return pano, used, st, "ALL"

    # 2) try dropping ONE image at a time
    for i in range(len(files)):
        use_files = files[:i] + files[i+1:]
        pano2, used2, st2 = attempt(use_files)
        if pano2 is not None:
            return pano2, used2, st2, f"DROPPED {files[i].name}"

    return None, [], best_status, "FAILED"

def main():
    script_dir = Path(__file__).resolve().parent
    images_dir = script_dir / "images"
    out_path = script_dir / "ai-panorama.jpg"

    files = collect_image_files(images_dir)
    print(f"Using images from: {images_dir}")
    print(f"Found {len(files)} files:")
    for f in files:
        print(" ", f.name)

    if len(files) < 2:
        print("Need at least 2 images in ai-panorama/images/")
        return

    # Try a couple scales: smaller can stitch more
    for scale in (0.6, 0.5, 0.4):
        print(f"\n--- Trying scale={scale} ---")
        pano, used_files, status, info = try_sets(files, scale=scale)
        if pano is not None:
            cv.imwrite(str(out_path), pano)
            print(f"Saved: {out_path}")
            print(f"Stitched {len(used_files)} images ({info})")
            print("Used:")
            for f in used_files:
                print(" ", f.name)
            return

    print("Stitch failed at all scales.")
    print("If IMG_5837 won't connect, you're likely missing IMG_5838 (bridge image).")

if __name__ == "__main__":
    main()

