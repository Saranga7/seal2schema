from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

def standardize_schema(
    image_path,
    output_path,
    canvas_size=224,
    target_long_side=200,
    threshold=128
):
    # Load grayscale
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    # Binarize: white foreground on black background
    arr = (arr > threshold).astype(np.uint8) * 255

    # Find bounding box of foreground
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        # Empty image fallback
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        Image.fromarray(canvas).save(output_path)
        return

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = arr[y_min:y_max + 1, x_min:x_max + 1]

    # Resize with aspect ratio preserved
    h, w = cropped.shape
    if h >= w:
        new_h = target_long_side
        new_w = max(1, int(w * target_long_side / h))
    else:
        new_w = target_long_side
        new_h = max(1, int(h * target_long_side / w))

    cropped_img = Image.fromarray(cropped)
    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.NEAREST)

    # Paste into centered black canvas
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    x_offset = (canvas_size - new_w) // 2
    y_offset = (canvas_size - new_h) // 2
    canvas.paste(resized_img, (x_offset, y_offset))

    canvas.save(output_path)

def main():
    input_dir = Path("/scratch/mahantas/datasets/MonogramSchema_Seal_pairs/schemas")
    output_dir = Path("/scratch/mahantas/datasets/MonogramSchema_Seal_pairs/schemas_standardized_224")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(input_dir.glob("*.png")):
        out_path = output_dir / img_path.name
        standardize_schema(img_path, out_path)

if __name__ == "__main__":
    main()