import cv2
import os
import matplotlib.pyplot as plt

def preprocess_and_save_masks(mask_dirs):
    for mask_dir in mask_dirs:
        for filename in os.listdir(mask_dir):
            file_path = os.path.join(mask_dir, filename)
            mask = cv2.imread(file_path)
            if mask is None:
                print(f"⚠️ Skipping {filename}, could not read file.")
                continue
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, binary_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            processed = cv2.dilate(binary_mask, kernel, iterations=1)
            cv2.imwrite(file_path, processed)
            print(f"✅ Processed: {filename}")

def preview_masks(mask_dirs, num_samples=3):
    sample_paths = []
    for mask_dir in mask_dirs:
        files = os.listdir(mask_dir)
        for f in files[:num_samples]:
            sample_paths.append(os.path.join(mask_dir, f))
    plt.figure(figsize=(10, 5))
    for i, path in enumerate(sample_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, len(sample_paths), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(os.path.basename(path))
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    mask_dirs = [
        r"healthy1",
        r"nitrogen1",
        r"pottasium1",
        r"sulpher1",
        r"zinc1"
    ]
    preprocess_and_save_masks(mask_dirs)
    preview_masks(mask_dirs)
