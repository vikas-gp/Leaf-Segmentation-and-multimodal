# predict.py (enhanced - final version without per-image JSON files)
import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage import binary_closing, binary_opening
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from skimage.filters import threshold_otsu

from model import UNet

# ---------- Config ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATHS = ["ckpt_unet_allself_bs8_s42.pth", "ckpt_unet_allself_bs8_s7.pth"]  # ensemble checkpoints
INFER_SIZE = (512, 512)
BASE_THRESH = 0.08
MIN_AREA = 80
OUTPUT_DIR = "output-K"
CONNECT_MAX_DIST = 40
CONNECT_MIN_PROB = 0.15

USE_TTA = True  # Test-Time Augmentation
USE_ENSEMBLE = True  # Use multiple checkpoints

SUBFOLDERS = {
    "confidence": "confidence",
    "mask": "mask",
    "skeleton": "skeleton",
    "veins_rgb": "veins_rgb",
    "overlay": "overlay"
}
for sf in SUBFOLDERS.values():
    os.makedirs(os.path.join(OUTPUT_DIR, sf), exist_ok=True)


# ---------- Load models ----------
def load_models(ckpt_paths=CKPT_PATHS, device=DEVICE):
    """Load multiple checkpoints for ensemble"""
    models = []
    for ckpt in ckpt_paths:
        if os.path.exists(ckpt):
            model = UNet().to(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
            print(f"✅ Loaded {ckpt}")
        else:
            print(f"⚠️  Checkpoint not found: {ckpt}")

    if len(models) == 0:
        raise RuntimeError("No valid checkpoints found!")
    return models


# ---------- Preprocess ----------
to_tensor_transform = transforms.Compose([
    transforms.Resize(INFER_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ---------- CLAHE ----------
def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------- TTA Prediction ----------
@torch.no_grad()
def predict_with_tta(models, inp, device=DEVICE, use_tta=USE_TTA, use_ensemble=USE_ENSEMBLE):
    """
    Ensemble prediction with Test-Time Augmentation
    - Average predictions from multiple models
    - Average predictions from flipped versions (TTA)
    """
    logits_sum = torch.zeros(1, 1, inp.shape[2], inp.shape[3], device=device)
    count = 0

    # Select models for ensemble
    model_list = models if use_ensemble else [models[0]]

    for model in model_list:
        # Original
        out = model(inp)
        logits = out[0] if isinstance(out, (list, tuple)) else out
        logits_sum += logits
        count += 1

        if use_tta:
            # Horizontal flip
            inp_hflip = torch.flip(inp, [3])
            out_hflip = model(inp_hflip)
            logits_hflip = out_hflip[0] if isinstance(out_hflip, (list, tuple)) else out_hflip
            logits_sum += torch.flip(logits_hflip, [3])
            count += 1

            # Vertical flip
            inp_vflip = torch.flip(inp, [2])
            out_vflip = model(inp_vflip)
            logits_vflip = out_vflip[0] if isinstance(out_vflip, (list, tuple)) else out_vflip
            logits_sum += torch.flip(logits_vflip, [2])
            count += 1

    return logits_sum / count


# ---------- Adaptive Thresholding ----------
def adaptive_threshold(probs, base_thresh=BASE_THRESH):
    """Use Otsu's method but ensure minimum threshold"""
    try:
        thresh_otsu = threshold_otsu(probs)
        return max(thresh_otsu, base_thresh)
    except:
        return base_thresh


# ---------- Adaptive Morphology ----------
def adaptive_morphology(mask, vein_density):
    """Adjust morphology kernel size based on vein density"""
    if vein_density < 0.05:  # sparse veins
        kernel_size = 3
        close_iter = 1
        open_iter = 1
    elif vein_density < 0.15:  # medium density
        kernel_size = 3
        close_iter = 2
        open_iter = 1
    else:  # dense veins
        kernel_size = 5
        close_iter = 2
        open_iter = 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return mask.astype(np.uint8)


# ---------- Enhanced Gap Connection ----------
def connect_gaps_improved(prob_map, mask, max_dist=CONNECT_MAX_DIST, min_prob=CONNECT_MIN_PROB, conn_dilate=2):
    """
    Enhanced gap connection with probability-based filtering
    Only connects endpoints if the path has reasonable average probability
    """
    bin_mask = prob_map > 0.05
    sk = skeletonize(bin_mask).astype(np.uint8)
    if sk.sum() == 0:
        return mask.astype(np.uint8)

    cost = 1.0 / np.clip(prob_map, 0.01, 1.0)

    kernel = np.ones((3, 3), np.uint8)
    neigh_count = cv2.filter2D(sk, -1, kernel)
    endpoints = np.argwhere((sk == 1) & (neigh_count <= 2))

    if len(endpoints) < 2:
        return mask.astype(np.uint8)

    paths_mask = np.zeros_like(sk, dtype=np.uint8)

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            p1 = tuple(endpoints[i])
            p2 = tuple(endpoints[j])
            dist = np.linalg.norm(np.array(p1) - np.array(p2))

            if dist <= max_dist:
                try:
                    _, path = route_through_array(cost, p1, p2, fully_connected=True)
                    path_probs = [prob_map[r, c] for r, c in path]
                    if np.mean(path_probs) >= min_prob:
                        for r, c in path:
                            paths_mask[r, c] = 1
                except Exception:
                    pass

    if paths_mask.sum() > 0:
        paths_mask = cv2.dilate(paths_mask, np.ones((conn_dilate, conn_dilate), np.uint8), iterations=1)

    merged = np.clip(mask.astype(np.uint8) + paths_mask, 0, 1)
    return merged.astype(np.uint8)


# ---------- Vein Classification and Differential Dilation ----------
def classify_and_dilate_veins(mask, skeleton):
    """
    Classify veins into primary/secondary based on thickness
    Apply differential dilation
    """
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    primary_mask = (dist_transform > 3).astype(np.uint8)
    secondary_mask = ((dist_transform > 1) & (dist_transform <= 3)).astype(np.uint8)
    tertiary_mask = ((dist_transform > 0) & (dist_transform <= 1)).astype(np.uint8)

    primary_dilated = cv2.dilate(primary_mask, np.ones((5, 5), np.uint8), iterations=1)
    secondary_dilated = cv2.dilate(secondary_mask, np.ones((3, 3), np.uint8), iterations=1)
    tertiary_dilated = tertiary_mask

    final_mask = np.clip(primary_dilated + secondary_dilated + tertiary_dilated, 0, 1)
    return final_mask.astype(np.uint8)


# ---------- Compute Metrics ----------
def compute_metrics(mask, skeleton, image_name):
    """Compute vein segmentation metrics"""
    total_vein_pixels = int(mask.sum())
    skeleton_length = int(skeleton.sum())
    avg_width = float(total_vein_pixels / max(skeleton_length, 1))

    num_labels, _ = cv2.connectedComponents(mask)
    num_segments = num_labels - 1

    metrics = {
        "image": image_name,
        "total_vein_pixels": total_vein_pixels,
        "skeleton_length": skeleton_length,
        "avg_vein_width": round(avg_width, 2),
        "num_vein_segments": num_segments,
        "vein_density": round(mask.mean(), 4)
    }

    return metrics


# ---------- Refine Thick Mask ----------
def refine_thick_mask(mask, min_area=MIN_AREA):
    """Remove tiny components"""
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    return cleaned.astype(np.uint8)


# ---------- Main Prediction ----------
@torch.no_grad()
def predict_and_save(models, image_path, save_dir=OUTPUT_DIR,
                     base_thresh=BASE_THRESH, min_area=MIN_AREA,
                     save_overlay=True):
    base = os.path.splitext(os.path.basename(image_path))[0]

    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    img_np_rgb = np.array(pil_img)
    img_np_rgb = apply_clahe(img_np_rgb)
    pil_img = Image.fromarray(img_np_rgb)

    inp = to_tensor_transform(pil_img).unsqueeze(0).to(DEVICE)

    logits_avg = predict_with_tta(models, inp, device=DEVICE,
                                  use_tta=USE_TTA, use_ensemble=USE_ENSEMBLE)

    probs_small = torch.sigmoid(logits_avg)[0, 0].cpu().numpy()
    probs = cv2.resize(probs_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    conf_0_255 = (np.clip(probs, 0, 1) * 255).astype(np.uint8)
    conf_path = os.path.join(save_dir, SUBFOLDERS["confidence"], f"{base}_confidence.png")
    cv2.imwrite(conf_path, conf_0_255)

    thresh = adaptive_threshold(probs, base_thresh)
    bin_mask = (probs > thresh).astype(np.uint8)

    vein_density = bin_mask.mean()

    refined = refine_thick_mask(bin_mask, min_area=min_area)
    refined = adaptive_morphology(refined, vein_density)

    connected = connect_gaps_improved(probs, refined, max_dist=CONNECT_MAX_DIST,
                                      min_prob=CONNECT_MIN_PROB, conn_dilate=2)

    final_mask = refine_thick_mask(connected, min_area=min_area)

    skeleton = skeletonize(final_mask > 0).astype(np.uint8)
    final_mask = classify_and_dilate_veins(final_mask, skeleton)

    skeleton = skeletonize(final_mask > 0).astype(np.uint8)

    metrics = compute_metrics(final_mask, skeleton, base)
    print(f"{base} | thresh={thresh:.3f} | density={metrics['vein_density']:.4f} | "
          f"skeleton_len={metrics['skeleton_length']} | segments={metrics['num_vein_segments']}")

    veins_white = (final_mask * 255).astype(np.uint8)
    mask_path = os.path.join(save_dir, SUBFOLDERS["mask"], f"{base}_mask.png")
    cv2.imwrite(mask_path, veins_white)

    sk_path = os.path.join(save_dir, SUBFOLDERS["skeleton"], f"{base}_skeleton.png")
    cv2.imwrite(sk_path, (skeleton * 255).astype(np.uint8))

    veins_rgb = np.zeros_like(img_np_rgb)
    veins_rgb[final_mask == 1] = img_np_rgb[final_mask == 1]
    veins_rgb_path = os.path.join(save_dir, SUBFOLDERS["veins_rgb"], f"{base}_veins_rgb.png")
    cv2.imwrite(veins_rgb_path, cv2.cvtColor(veins_rgb, cv2.COLOR_RGB2BGR))

    overlay_path = None
    if save_overlay:
        contours, _ = cv2.findContours(veins_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        overlay_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)
        overlay_path = os.path.join(save_dir, SUBFOLDERS["overlay"], f"{base}_overlay.png")
        cv2.imwrite(overlay_path, overlay_bgr)

    return {
        "mask_path": mask_path,
        "skeleton_path": sk_path,
        "veins_rgb_path": veins_rgb_path,
        "confidence_path": conf_path,
        "overlay_path": overlay_path,
        "metrics": metrics
    }


# ---------- Main ----------
if __name__ == "__main__":
    test_dir = "test K"
    exts = [".jpg", ".jpeg", ".png"]

    models = load_models(CKPT_PATHS)
    print(f"Loaded {len(models)} model(s) for {'ensemble' if USE_ENSEMBLE else 'single'} prediction")
    print(f"TTA: {'Enabled' if USE_TTA else 'Disabled'}")

    all_metrics = []

    for fname in sorted(os.listdir(test_dir)):
        if any(fname.lower().endswith(ext) for ext in exts):
            image_path = os.path.join(test_dir, fname)
            print(f"\n🌿 Processing: {fname}")
            result = predict_and_save(models, image_path,
                                      base_thresh=BASE_THRESH,
                                      min_area=MIN_AREA,
                                      save_overlay=True)
            all_metrics.append(result["metrics"])

    # Save only ONE summary JSON with all metrics
    summary_path = os.path.join(OUTPUT_DIR, "summary_metrics.json")
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✅ Done! Results saved to: {OUTPUT_DIR}")
    print(f"📊 Summary metrics: {summary_path}")
