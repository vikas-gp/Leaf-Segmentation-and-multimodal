import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage import binary_closing, binary_opening
from skimage.morphology import skeletonize
from skimage.graph import route_through_array

from model import UNet

# ---------- Config ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "ckpt_unet_allself_bs8_s42.pth"   # adjust if needed
INFER_SIZE = (512, 512)
THRESH = 0.08             # gentler threshold
MIN_AREA = 80              # remove tiny blobs
OUTPUT_DIR = "output-old"
CONNECT_THRESH = 0.08
CONNECT_MAX_DIST = 40

SUBFOLDERS = {
    "confidence": "confidence",
    "mask": "mask",            # thick final mask
    "skeleton": "skeleton",    # optional thin skeleton for analysis
    "veins_rgb": "veins_rgb",
    "overlay": "overlay"
}
for sf in SUBFOLDERS.values():
    os.makedirs(os.path.join(OUTPUT_DIR, sf), exist_ok=True)

# ---------- Load model once ----------
def load_model(ckpt=CKPT_PATH, device=DEVICE):
    model = UNet().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ---------- Preprocess ----------
to_tensor_256 = transforms.Compose([
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

# ---------- connector (Dijkstra) ----------
def connect_gaps_to_mask(prob_map, mask, thresh=CONNECT_THRESH, max_dist=CONNECT_MAX_DIST, conn_dilate=2):
    """
    Finds skeleton endpoints from prob_map>thresh and connects close pairs by Dijkstra on cost array.
    Adds the path to `mask`, dilates paths to keep thickness.
    Returns updated mask (binary).
    """
    bin_mask = prob_map > thresh
    sk = skeletonize(bin_mask).astype(np.uint8)
    if sk.sum() == 0:
        return mask.astype(np.uint8)

    cost = -np.log(np.clip(prob_map, 1e-6, 1.0))
    kernel = np.ones((3, 3), np.uint8)
    neigh_count = cv2.filter2D(sk, -1, kernel)
    endpoints = np.argwhere((sk == 1) & (neigh_count <= 2))

    # if too few endpoints, quick return
    if len(endpoints) < 2:
        return mask.astype(np.uint8)

    # make a copy to draw paths
    paths_mask = np.zeros_like(sk, dtype=np.uint8)

    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            p1 = tuple(endpoints[i])
            p2 = tuple(endpoints[j])
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist <= max_dist:
                try:
                    _, path = route_through_array(cost, p1, p2, fully_connected=True)
                    for r, c in path:
                        paths_mask[r, c] = 1
                except Exception:
                    pass

    # dilate connector paths for thickness (so they become visible primary/secondary veins)
    if paths_mask.sum() > 0:
        paths_mask = cv2.dilate(paths_mask.astype(np.uint8), np.ones((conn_dilate, conn_dilate), np.uint8), iterations=1)

    # merge connectors into mask and return
    merged = np.clip(mask.astype(np.uint8) + paths_mask, 0, 1)
    return merged.astype(np.uint8)

# ---------- refine mask (thick) ----------
def refine_thick_mask(mask, min_area=MIN_AREA, close_iter=1):
    """
    Keep a thick mask:
    - remove tiny components
    - mild morphological close/open to smooth while preserving width
    """
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    # small close to fill tiny gaps; keep kernel small so primaries are preserved
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=close_iter)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    return cleaned.astype(np.uint8)

# ---------- prediction ----------
@torch.no_grad()
def predict_and_save(model, image_path, save_dir=OUTPUT_DIR,
                     thresh=THRESH, min_area=MIN_AREA,
                     connect_thresh=CONNECT_THRESH, connect_max_dist=CONNECT_MAX_DIST,
                     save_overlay=True):
    base = os.path.splitext(os.path.basename(image_path))[0]

    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    img_np_rgb = np.array(pil_img)
    img_np_rgb = apply_clahe(img_np_rgb)
    pil_img = Image.fromarray(img_np_rgb)

    inp = to_tensor_256(pil_img).unsqueeze(0).to(DEVICE)

    out = model(inp)
    # robust unpack: model might return (mask, sk) or a single tensor
    if isinstance(out, (list, tuple)):
        if len(out) >= 1:
            logits = out[0]
        else:
            raise RuntimeError("model returned empty tuple")
    else:
        logits = out

    probs_small = torch.sigmoid(logits)[0, 0].cpu().numpy()
    probs = cv2.resize(probs_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # save confidence map
    conf_0_255 = (np.clip(probs, 0, 1) * 255).astype(np.uint8)
    conf_path = os.path.join(save_dir, SUBFOLDERS["confidence"], f"{base}_confidence.png")
    cv2.imwrite(conf_path, conf_0_255)

    # coarse binary mask
    bin_mask = (probs > thresh).astype(np.uint8)

    # first refine to keep thick veins
    thick_refined = refine_thick_mask(bin_mask, min_area=min_area, close_iter=1)

    # connect gaps (adds dilated connector paths)
    thick_connected = connect_gaps_to_mask(probs, thick_refined, thresh=connect_thresh, max_dist=connect_max_dist, conn_dilate=2)

    # final mild clean
    final_thick = refine_thick_mask(thick_connected, min_area=min_area, close_iter=0)

    pos_ratio = float(final_thick.mean())
    print(f"{base} | thresh={thresh:.3f} | pos_ratio={pos_ratio:.4f} | prob min/max={probs.min():.3f}/{probs.max():.3f}")

    # save thick mask (white)
    veins_white = (final_thick * 255).astype(np.uint8)
    mask_path = os.path.join(save_dir, SUBFOLDERS["mask"], f"{base}_thick_mask.png")
    cv2.imwrite(mask_path, veins_white)

    # save optional skeleton (thin) for analysis only
    skeleton = skeletonize(final_thick > 0).astype(np.uint8)
    sk_path = os.path.join(save_dir, SUBFOLDERS["skeleton"], f"{base}_skeleton.png")
    cv2.imwrite(sk_path, (skeleton * 255).astype(np.uint8))

    # veins rgb for visualization (copy original colors where mask==1)
    veins_rgb = np.zeros_like(img_np_rgb)
    veins_rgb[final_thick == 1] = img_np_rgb[final_thick == 1]
    veins_rgb_path = os.path.join(save_dir, SUBFOLDERS["veins_rgb"], f"{base}_veins_rgb.png")
    cv2.imwrite(veins_rgb_path, cv2.cvtColor(veins_rgb, cv2.COLOR_RGB2BGR))

    # overlay
    overlay_path = None
    if save_overlay:
        contours, _ = cv2.findContours(veins_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        overlay_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)  # thicker overlay line
        overlay_path = os.path.join(save_dir, SUBFOLDERS["overlay"], f"{base}_overlay.png")
        cv2.imwrite(overlay_path, overlay_bgr)

    return {
        "mask_path": mask_path,
        "skeleton_path": sk_path,
        "veins_rgb_path": veins_rgb_path,
        "confidence_path": conf_path,
        "overlay_path": overlay_path,
        "pos_ratio": pos_ratio
    }

# ---------- main ----------
if __name__ == "__main__":
    test_dir = "test1"
    exts = [".jpg", ".jpeg", ".png"]

    model = load_model()

    for fname in sorted(os.listdir(test_dir)):
        if any(fname.lower().endswith(ext) for ext in exts):
            image_path = os.path.join(test_dir, fname)
            print("Processing", fname)
            predict_and_save(model, image_path, thresh=THRESH, min_area=MIN_AREA,
                             connect_thresh=CONNECT_THRESH, connect_max_dist=CONNECT_MAX_DIST,
                             save_overlay=True)

    print("Done. Results:", OUTPUT_DIR)