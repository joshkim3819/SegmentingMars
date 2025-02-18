import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Part 1

#Adjusting the lighting, tried using CLAHE to clarify images
def adjust_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    adjusted = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    adjusted = cv2.GaussianBlur(adjusted, (3, 3), 0)
    return adjusted

# Using IoU (SAM Metric) to identify overlaps of SAM's masks and Grounding DINO's label boxes
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea <= 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)

# Showing the masks from SAM
def visualize_masks(image, masks):
    vis = image.copy()
    for m in masks:
        seg = m['segmentation']
        seg_uint8 = (seg.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        x, y, w, h = [int(v) for v in m['bbox']]
        cv2.putText(vis, m.get('label', 'unknown'), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis

# Part 2

# Setting up SAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
sam_model_type = "vit_h"
sam_checkpoint = "models/sam_vit_h_4b8939.pth"
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)

# Setting up Grounding Dino
from groundingdino.util.inference import load_model, predict
gdino_config_path = "models/groundingdino/config.py"
gdino_checkpoint = "models/groundingdino/groundingdino_swint_ogc.pth" 
device = "cuda" if torch.cuda.is_available() else "cpu"
gdino_model = load_model(gdino_config_path, gdino_checkpoint, device=device)

# Part 3
# Making SAM create masks
def get_segmentation_masks(image):
    masks = mask_generator.generate(image)
    return [m for m in masks]

# Making Grounding DINO make labels
def run_grounding_dino(pil_image, prompt, model, box_threshold=0.3, text_threshold=0.25):
    boxes, logits, phrases = predict(model, pil_image, caption=prompt,
                                     box_threshold=box_threshold,
                                     text_threshold=text_threshold)
    return boxes, phrases

# Combining outputs from SAM and Grounding DINO using IoU from beginning of file
def label_masks_with_grounding_dino(masks, gdino_boxes, gdino_phrases, iou_threshold=0.3):
    labeled = []
    for mask in masks:
        x, y, w, h = mask['bbox']
        seg_bbox = [x, y, x + w, y + h]
        best_iou = 0
        best_label = "unknown"
        for box, phrase in zip(gdino_boxes, gdino_phrases):
            iou = compute_iou(seg_bbox, box)
            if iou > best_iou:
                best_iou = iou
                best_label = phrase
        mask['label'] = best_label if best_iou > iou_threshold else "unknown"
        labeled.append(mask)
    return labeled

# Attempt at fine tuning and creating final outputs
def main(image_path, output_path="output.png"):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Cannot load image", image_path)
        return
    image = cv2.imread(image_path)

    adjusted = adjust_lighting(image)
    masks = get_segmentation_masks(adjusted)
    print("SAM produced", len(masks), "masks for", image_path)

    rgb = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(rgb)

    prompt = "rock, crater, soil, dune, mountain, sky"
    gdino_boxes, gdino_phrases = run_grounding_dino(pil_im, prompt, gdino_model)
    print("Grounding Dino found", len(gdino_boxes), "objects for", image_path)

    labeled = label_masks_with_grounding_dino(masks, gdino_boxes, gdino_phrases)
    result = visualize_masks(adjusted, labeled)
    cv2.imwrite(output_path, result)
    print("Saved output to", output_path)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Logistics for downloading the file
def process_images(input_folder="data", output_folder="output"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            image_path = os.path.join(input_folder, file)
            out_path = os.path.join(output_folder, file)
            print("\nProcessing", image_path)
            main(image_path, out_path)

# Saw this on a Medium article, so I wanted to try it out
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("No image provided; processing images from 'data' folder.")
        process_images("data", "output")
    else:
        path = sys.argv[1]
        if os.path.isdir(path):
            process_images(path, "output")
        else:
            if not os.path.exists("output"):
                os.makedirs("output")
            main(path, os.path.join("output", os.path.basename(path)))