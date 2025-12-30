import os
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import json

load_dotenv()
token = os.getenv("HF_TOKEN")

from transformers import Sam3Processor, Sam3Model

CLASS_PROMPT = "pig"
CLASS_ID = 1
CONFIDENCE_THRESHOLD = 0.4
SOURCE_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife"
OUTPUT_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/teacher"

def generate_predictions(subset_name, model, processor, device):
    image_dir = os.path.join(SOURCE_ROOT, subset_name, "images")

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    coco_results = []

    for img_name in tqdm(image_files):
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        inputs = processor(images=image, text=CLASS_PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_instance_segmentation(
            outputs, 
            threshold=CONFIDENCE_THRESHOLD, 
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        image_id = os.path.splitext(img_name)[0]

        for box, score in zip(boxes, scores):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            coco_results.append({
                "image_id": image_id,
                "category_id": CLASS_ID,
                "bbox": [round(float(x_min), 2), 
                         round(float(y_min), 2), 
                         round(float(width), 2), 
                         round(float(height), 2)],
                "score": round(float(score), 4)
            })

    output_file = os.path.join(OUTPUT_ROOT, f"predictions_2.json")
    with open(output_file, "w") as f:
        json.dump(coco_results, f)
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    generate_predictions("test", model, processor, device)