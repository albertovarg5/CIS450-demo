import cv2 as cv
import sys
import os

def stitch_images(images):
    stitcher = cv.Stitcher_create(cv.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images)

    if status != cv.Stitcher_OK:
        print("Panorama stitching failed")
        sys.exit(1)

    return pano


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python panorama.py panorama image1 image2 ...")
        sys.exit(1)

    output_dir = sys.argv[1]
    image_paths = sys.argv[2:]

    images = []
    for path in image_paths:
        img = cv.imread(path)
        if img is None:
            print(f"Could not read image {path}")
            sys.exit(1)
        images.append(img)

    pano = stitch_images(images)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "result.jpg")
    cv.imwrite(output_path, pano)

    print(f"Panorama saved to {output_path}")
