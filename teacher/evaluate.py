import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# MODEL = "yolov8m"
GT_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife"
PRED_ROOT = f"/hd2/marcos/research/repos/pig-segmentation-distill/data/SAM3_PigLife/test/labels"
OUTPUT_METRICS_PATH = f"/hd2/marcos/research/repos/pig-segmentation-distill/teacher"
SUBSET = "test"

def read_yolo_file(file_path, w, h, is_gt=False):
    boxes = []
    labels = []
    scores = []
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = list(map(float, line.strip().split()))
            
            if len(parts) < 5: 
                continue

            cls_id = int(parts[0])
            x_c, y_c, bw, bh = parts[1:5]
            
            if is_gt:
                conf = 1.0
            else:
                conf = parts[5] if len(parts) > 5 else 1.0
            
            x_min = (x_c - bw / 2) * w
            y_min = (y_c - bh / 2) * h
            x_max = (x_c + bw / 2) * w
            y_max = (y_c + bh / 2) * h
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(cls_id)
            scores.append(conf)
            
    if len(boxes) == 0:
        return (
            torch.empty((0, 4), dtype=torch.float32), 
            torch.empty((0,), dtype=torch.int64), 
            torch.empty((0,), dtype=torch.float32)
        )
        
    return (
        torch.tensor(boxes, dtype=torch.float32), 
        torch.tensor(labels, dtype=torch.int64), 
        torch.tensor(scores, dtype=torch.float32)
    )

def evaluate_subset(subset_name):
    gt_img_dir = os.path.join(GT_ROOT, subset_name, "images")
    gt_lbl_dir = os.path.join(GT_ROOT, subset_name, "labels")

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)
    
    image_files = [f for f in os.listdir(gt_img_dir) if f.lower().endswith(".jpg")]
    
    for img_file in tqdm(image_files):
        file_id = os.path.splitext(img_file)[0]
        txt_file = file_id + ".txt"
        
        img_path = os.path.join(gt_img_dir, img_file)
        with Image.open(img_path) as img:
                w, h = img.size
            
        gt_path = os.path.join(gt_lbl_dir, txt_file)
        gt_boxes, gt_labels, _ = read_yolo_file(gt_path, w, h, is_gt=True)
        
        pred_path = os.path.join(PRED_ROOT, txt_file)
        p_boxes, p_labels, p_scores = read_yolo_file(pred_path, w, h, is_gt=False)
        
        metric.update(
            preds=[{
                "boxes": p_boxes,
                "scores": p_scores,
                "labels": p_labels
            }],
            target=[{
                "boxes": gt_boxes,
                "labels": gt_labels
            }]
        )

    result = metric.compute()

    metrics = [{
        "Model": "zero-shot SAM3",
        "mAP_50-95": result['map'].item(),
        "mAP_50": result['map_50'].item(),
        "mAP_75": result['map_75'].item(),
        "AP_Medium": result['map_medium'].item(),
        "AP_Large": result['map_large'].item()
    }]

    os.makedirs(OUTPUT_METRICS_PATH, exist_ok=True)
    df = pd.DataFrame(metrics)
    print(df)
    df.to_csv(f"{OUTPUT_METRICS_PATH}/zero-shot_SAM3_performance.csv", index=False)

if __name__ == "__main__":
    evaluate_subset(SUBSET)