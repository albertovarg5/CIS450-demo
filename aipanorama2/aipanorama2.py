import cv2 as cv
import glob

print("Loading images...")

images = []
filenames = sorted(glob.glob("*.png"))

for filename in filenames:
    print("Reading:", filename)
    img = cv.imread(filename)
    if img is not None:
        images.append(img)

print(f"Total images loaded: {len(images)}")

if len(images) < 2:
    print("Not enough images to stitch.")
    exit()

print("Creating panorama...")

stitcher = cv.Stitcher_create()
status, panorama = stitcher.stitch(images)

if status == cv.Stitcher_OK:
    cv.imwrite("aipanorama2.jpg", panorama)
    print("Panorama created successfully: aipanorama2.jpg")
else:
    print("Panorama stitching failed. Status:", status)
