import os
import random
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision import transforms

class PairedAugmentation:
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)

    def __call__(self, img: Image.Image, mask: Image.Image):
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        angle = random.uniform(-10, 10)
        img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
        img = self.color_jitter(img)
        return img, mask

def augment_dataset_inplace(img_dir, mask_dir, augmentations_per_image=2):
    aug = PairedAugmentation()
    img_files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])

    for img_file in img_files:
        base, ext = os.path.splitext(img_file)
        mask_file = None
        for e in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            if os.path.exists(os.path.join(mask_dir, base + e)):
                mask_file = base + e
                break
        if mask_file is None:
            print(f"No mask found for {img_file}, skipping.")
            continue

        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        for i in range(1, augmentations_per_image + 1):
            aug_img, aug_mask = aug(image, mask)
            aug_img_name = f"{base}_aug{i}{ext}"
            aug_mask_name = f"{base}_aug{i}{ext}"
            aug_img.save(os.path.join(img_dir, aug_img_name))
            aug_mask.save(os.path.join(mask_dir, aug_mask_name))
            print(f"Saved augmented: {aug_img_name} and {aug_mask_name}")

if __name__ == "__main__":
    # Set your original folders here
    train_img_dir = 'new d/images'
    train_mask_dir = 'new d/veins'
    #val_img_dir = 'dataset/val/images'
    #val_mask_dir = 'dataset/val/masks'

    aug_per_image = 2

    print("Augmenting TRAIN dataset in-place...")
    augment_dataset_inplace(train_img_dir, train_mask_dir, aug_per_image)


    print("Augmentation complete.")
