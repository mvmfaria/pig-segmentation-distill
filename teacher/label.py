import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv() 

token = os.getenv("HF_TOKEN")

from transformers import Sam3Processor, Sam3Model

SOURCE_IMAGE_DIR = "/hd2/marcos/research/repos/cross-domain-pig-detection/data/PigLife/val/images"
OUTPUT_DATA_DIR = "/hd2/marcos/research/repos/pig-segmentation-distill/data/SAM3_PigLife"
CLASS_PROMPT = "pig"
CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.4

def mask_to_yolo_polygon(binary_mask, width, height):
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 50: 
            continue
            
        flattened = contour.flatten().tolist()
        normalized_poly = []
        
        for i in range(0, len(flattened), 2):
            x = flattened[i]
            y = flattened[i+1]
            normalized_poly.append(min(max(x / width, 0.0), 1.0))
            normalized_poly.append(min(max(y / height, 0.0), 1.0))
            
        if len(normalized_poly) > 4:
            polygons.append(normalized_poly)
            
    return polygons

def label():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    os.makedirs(f"{OUTPUT_DATA_DIR}/val/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DATA_DIR}/val/labels", exist_ok=True)
    
    image_files = [f for f in os.listdir(SOURCE_IMAGE_DIR) if f.lower().endswith(".jpg")]
    print(f"Found {len(image_files)} images.")
    
    for img_name in tqdm(image_files, desc="Distilling Knowledge"):
        img_path = os.path.join(SOURCE_IMAGE_DIR, img_name)
        
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        inputs = processor(
            images=image, 
            text=CLASS_PROMPT, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_instance_segmentation(
            outputs, 
            threshold=CONFIDENCE_THRESHOLD, 
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results["masks"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        yolo_lines = []
        
        for i, mask in enumerate(masks):
            if scores[i] < CONFIDENCE_THRESHOLD:
                continue

            polygons = mask_to_yolo_polygon(mask, w, h)
            
            for poly in polygons:
                poly_str = " ".join([f"{coord:.6f}" for coord in poly])
                line = f"{CLASS_ID} {poly_str}"
                yolo_lines.append(line)
        
        if yolo_lines:
            out_img_path = f"{OUTPUT_DATA_DIR}/val/images/{img_name}"
            image.save(out_img_path)
            
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            out_txt_path = f"{OUTPUT_DATA_DIR}/val/labels/{txt_name}"
            with open(out_txt_path, "w") as f:
                f.write("\n".join(yolo_lines))

if __name__ == "__main__":
    label()