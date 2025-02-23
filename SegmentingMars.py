import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from groundingdino.util.inference import load_model, predict

# Loading up the models
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth").to(device)
mask_gen = SamAutomaticMaskGenerator(sam)
gdino = load_model("models/groundingdino/GroundingDINO_SwinT_OGC.yaml", 
                   "models/groundingdino/groundingdino_swint_ogc.pth", device=device)

# Applying CLAHE to improve image quality for better segmentation
def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    return cv2.GaussianBlur(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), (3, 3), 0)

# Convert Grounding DINO label boxes to coordinates
def gdino_to_bbox(box, w, h):
    xc, yc, bw, bh = box
    return [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]

# Calculating Intersection over Union between the two boxes
def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if not inter:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)

# Display the masks
def draw_masks(img, masks):
    vis = img.copy()
    for mask in masks:
        seg = (mask["segmentation"].astype(np.uint8) * 255)
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        x, y, w, h = [int(v) for v in mask["bbox"]]
        label = mask.get("label", "unknown")
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis

# Trying out SAM segmentation with Grounding DINO Label
def segment_and_label(img_path, out_path="output.png", display=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        return False

    enhanced = enhance_image(img)
    masks = mask_gen.generate(enhanced)
    print(f"Got {len(masks)} masks from SAM")

    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    h, w = enhanced.shape[:2]

    boxes, _, phrases = predict(model=gdino, image=pil_img, text_prompt="background, bedrock, martian soil, rocks, sands, tire tracks, unknown",
                                box_threshold=0.25, text_threshold=0.25)
    print(f"Grounding DINO detected {len(boxes)} objects")

    for mask in masks:
        x, y, w_mask, h_mask = mask["bbox"]
        mask_bbox = [x, y, x + w_mask, y + h_mask]
        best_iou, best_label = 0, "unknown"
        for box, phrase in zip(boxes, phrases):
            dino_bbox = gdino_to_bbox(box, w, h)
            score = iou(mask_bbox, dino_bbox)
            if score > best_iou:
                best_iou, best_label = score, phrase
        mask["label"] = best_label if best_iou > 0.3 else "unknown"

    result = draw_masks(enhanced, masks)
    cv2.imwrite(out_path, result)
    if display:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    return True

# Saving the produced images in a folder called "Output"
def process_folder(in_dir="data", out_dir="output"):
    if not os.path.exists(in_dir):
        print(f"Error: Input directory '{in_dir}' doesn't exist.")
        return
    os.makedirs(out_dir, exist_ok=True)
    processed = 0
    for f in os.listdir(in_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            in_path = os.path.join(in_dir, f)
            out_path = os.path.join(out_dir, f)
            print(f"Processing {in_path}")
            if segment_and_label(in_path, out_path, display=False):
                processed += 1
    print(f"Processed {processed} images.")

# In case there's any files that don't work (Purpose is for backup)
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        process_folder()
    else:
        path = sys.argv[1]
        if os.path.isdir(path):
            process_folder(path)
        elif os.path.isfile(path) and path.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            os.makedirs("output", exist_ok=True)
            segment_and_label(path, os.path.join("output", os.path.basename(path)))
        else:
            print(f"Error: '{path}' isn't a valid image file/directory.")
