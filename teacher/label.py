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

def label(output_data_dir, souce_image_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    image_files = [f for f in os.listdir(souce_image_dir) if f.lower().endswith(".jpg")]
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(souce_image_dir, img_name)
        
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
            line = f"{CLASS_ID} {box_str} {score:.6f}"
            yolo_lines.append(line)
        
        if yolo_lines:
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            out_txt_path = f"{output_data_dir}/{txt_name}"
            with open(out_txt_path, "w") as f:
                f.write("\n".join(yolo_lines))

if __name__ == "__main__":
    subsets = ["test", "train", "val"]

    for set in subsets:
        os.makedirs(f"/hd2/marcos/research/repos/pig-segmentation-distill/data/SAM3_PigLife_labels/{set}", exist_ok=True)
        souce_image_dir = f"/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife/{set}/images"
        output_data_dir = f"/hd2/marcos/research/repos/pig-segmentation-distill/data/SAM3_PigLife_labels/{set}" 
        label(output_data_dir, souce_image_dir)