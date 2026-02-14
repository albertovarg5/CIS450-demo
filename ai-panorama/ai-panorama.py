import cv2 as cv
from pathlib import Path


def load_images(folder: Path, max_width: int = 1200):
    paths = sorted(
        list(folder.glob("*.png")) +
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg"))
    )

    if len(paths) < 2:
        print("Error: Need at least 2 images in the images folder.")
        return [], []

    print("Image order used:")
    for p in paths:
        print(" ", p.name)

    images = []
    for p in paths:
        img = cv.imread(str(p))
        if img is None:
            print(f"Warning: could not read {p.name}")
            continue

        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv.resize(img, (int(w * scale), int(h * scale)))

        images.append(img)

    return paths, images


def stitch_two(a, b, mode):
    stitcher = cv.Stitcher_create(mode)

    # Lower threshold so it accepts weaker connections
    try:
        stitcher.setPanoConfidenceThresh(0.0)
    except Exception:
        pass

    status, pano = stitcher.stitch([a, b])
    return status, pano


def sequential_stitch(paths, images, mode, mode_name: str):
    print(f"\nTrying sequential stitch in {mode_name} mode...")
    pano = images[0]

    for i in range(1, len(images)):
        status, new_pano = stitch_two(pano, images[i], mode)

        if status != cv.Stitcher_OK:
            print(f"STOP: Failed when adding image #{i+1}: {paths[i].name} (status={status})")
            return None

        pano = new_pano
        print(f" OK: added image #{i+1} ({paths[i].name})")

    return pano


def main():
    images_dir = Path("images")
    if not images_dir.exists():
        print("Error: images folder not found.")
        return 1

    paths, images = load_images(images_dir, max_width=1200)
    if len(images) < 2:
        return 1

    print(f"\nLoaded {len(images)} image(s).")

    # Try PANORAMA first, then SCANS
    pano = sequential_stitch(paths, images, cv.Stitcher_PANORAMA, "PANORAMA")

    if pano is None:
        pano = sequential_stitch(paths, images, cv.Stitcher_SCANS, "SCANS")

    if pano is None:
        print("\nCould not stitch all 8 images together.")
        print("This almost always means parallax (camera moved forward/sideways) or not enough overlap.")
        print("Best fix: remove the problem image or retake photos with 30–50% overlap while rotating in place.")
        return 1

    # Add border (like the example) so the warped area is visible
    pano = cv.copyMakeBorder(pano, 200, 200, 200, 200, cv.BORDER_CONSTANT, value=(0, 0, 0))

    out_path = Path("ai-panorama.jpg")
    cv.imwrite(str(out_path), pano)
    print(f"\nSaved: {out_path}")
    print("Panorama complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






















