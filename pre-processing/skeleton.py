import os
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image

def process_folder(mask_dir):
    print(f"Processing: {mask_dir}")
    for fname in os.listdir(mask_dir):
        if fname.startswith('.') or not fname.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
            continue

        path = os.path.join(mask_dir, fname)
        img = Image.open(path).convert('L')
        arr = np.array(img) > 127   # binary mask
        sk = skeletonize(arr).astype(np.uint8) * 255

        base, _ = os.path.splitext(fname)
        out_path = os.path.join(mask_dir, f"{base}_sk.png")
        Image.fromarray(sk).save(out_path)
    print("Done.\n")

if __name__ == "__main__":
    process_folder("dataset/train/masks")
    process_folder("dataset/val/masks")
