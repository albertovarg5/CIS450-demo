from pathlib import Path
import cv2 as cv

def edge_detect(color_bgr, blur_ksize=5, t1=60, t2=160):

    gray = cv.cvtColor(color_bgr, cv.COLOR_BGR2GRAY)

    if blur_ksize and blur_ksize > 1:
        gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv.Canny(gray, t1, t2)
    return edges

def blend_edges_on_color(color_bgr, edges, alpha=0.85):
    edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    blended = cv.addWeighted(color_bgr, alpha, edges_bgr, 1 - alpha, 0)
    return blended

def main():
    edges_dir = Path(__file__).parent 
    image_exts = {".jpg", ".jpeg", ".png"}

    files = []
    for p in edges_dir.iterdir():
        if p.suffix.lower() in image_exts and not p.name.endswith(".edges.jpg") and not p.name.endswith(".blended.jpg"):
            files.append(p)

    if not files:
        print("No input images found in edges/ folder.")
        return

    blur_ksize = 5
    t1, t2 = 60, 160
    alpha = 0.85

    print(f"Found {len(files)} image(s). Using blur={blur_ksize}, Canny=({t1},{t2}), alpha={alpha}")

    for img_path in files:
        color = cv.imread(str(img_path))
        if color is None:
            print(f"Skipping (could not read): {img_path.name}")
            continue

        edges = edge_detect(color, blur_ksize=blur_ksize, t1=t1, t2=t2)

        edges_out = img_path.with_suffix("") 
        edges_out = edges_out.parent / f"{edges_out.name}.edges.jpg"
        cv.imwrite(str(edges_out), edges)
        blended = blend_edges_on_color(color, edges, alpha=alpha)
        blended_out = img_path.with_suffix("")
        blended_out = blended_out.parent / f"{blended_out.name}.blended.jpg"
        cv.imwrite(str(blended_out), blended)

        print(f"Saved: {edges_out.name} and {blended_out.name}")

if __name__ == "__main__":
    main()
