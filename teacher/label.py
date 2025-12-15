import os
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")

from transformers import Sam3Processor, Sam3Model

CLASS_PROMPT = "pig"
CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.4
SOURCE_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife"
OUTPUT_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/data/SAM3_PigLife"

def convert_box_to_yolo(box, img_width, img_height):
    x_min, y_min, x_max, y_max = box
    
    w_box = x_max - x_min
    h_box = y_max - y_min
    x_center = x_min + (w_box / 2)
    y_center = y_min + (h_box / 2)
    
    x_c_norm = x_center / img_width
    y_c_norm = y_center / img_height
    w_norm = w_box / img_width
    h_norm = h_box / img_height
    
    return [
        max(0.0, min(1.0, x_c_norm)),
        max(0.0, min(1.0, y_c_norm)),
        max(0.0, min(1.0, w_norm)),
        max(0.0, min(1.0, h_norm))
    ]

def create_output_dir(subset_name):
    out_dir = os.path.join(OUTPUT_ROOT, subset_name)
    os.makedirs(out_dir, exist_ok=True)

    dst_img_dir = os.path.join(OUTPUT_ROOT, subset_name, "images")
    dst_lbl_dir = os.path.join(OUTPUT_ROOT, subset_name, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

def label(subset_name, model, processor, device):
    create_output_dir(subset_name)
    
    image_files = [f for f in os.listdir(os.path.join(SOURCE_ROOT, subset_name, "images")) if f.lower().endswith(".jpg")]
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(SOURCE_ROOT, subset_name, "images", img_name)
        
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
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        yolo_lines = []
        
        for box, score in zip(boxes, scores):
            if score < CONFIDENCE_THRESHOLD:
                continue

            yolo_box = convert_box_to_yolo(box, w, h)
            
            box_str = " ".join([f"{coord:.6f}" for coord in yolo_box])
            if subset_name == "test":
                line = f"{CLASS_ID} {box_str} {score:.6f}"
            else:
                line = f"{CLASS_ID} {box_str}"
            yolo_lines.append(line)
        
        out_img_path = os.path.join(OUTPUT_ROOT, subset_name, "images", img_name)
        image.save(out_img_path)
        
        if yolo_lines:
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            out_txt_path = os.path.join(OUTPUT_ROOT, subset_name, "labels", txt_name)
            with open(out_txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
        else:
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            out_txt_path = os.path.join(OUTPUT_ROOT, subset_name, "labels", txt_name)
            open(out_txt_path, "w").close()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    subsets = ["test", "train", "val"]

    for set in subsets:
        label(set, model, processor, device)