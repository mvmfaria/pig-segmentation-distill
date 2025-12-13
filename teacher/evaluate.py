import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

GT_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife"
PRED_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/data/SAM3_PigLife_labels"
TEACHER_PATH = "/hd2/marcos/research/repos/pig-segmentation-distill/teacher"

SUBSET = "test"

def read_yolo_file(file_path, w, h, is_gt=False):
    boxes = []
    labels = []
    scores = []
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                
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
                
    return (
        torch.tensor(boxes, dtype=torch.float32), 
        torch.tensor(labels, dtype=torch.int64), 
        torch.tensor(scores, dtype=torch.float32)
    )

def evaluate_subset(subset_name):
    gt_img_dir = os.path.join(GT_ROOT, subset_name, "images")
    gt_lbl_dir = os.path.join(GT_ROOT, subset_name, "labels")
    pred_lbl_dir = os.path.join(PRED_ROOT, subset_name)

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    
    image_files = [f for f in os.listdir(gt_img_dir) if f.lower().endswith(".jpg")]
    
    preds = []
    targets = []
    
    for img_file in tqdm(image_files):
        file_id = os.path.splitext(img_file)[0]
        txt_file = file_id + ".txt"
        
        img_path = os.path.join(gt_img_dir, img_file)
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except:
            continue
            
        gt_path = os.path.join(gt_lbl_dir, txt_file)
        gt_boxes, gt_labels, _ = read_yolo_file(gt_path, w, h, is_gt=True)
        
        targets.append({
            "boxes": gt_boxes,
            "labels": gt_labels
        })
        
        pred_path = os.path.join(pred_lbl_dir, txt_file)
        p_boxes, p_labels, p_scores = read_yolo_file(pred_path, w, h, is_gt=False)
        
        preds.append({
            "boxes": p_boxes,
            "scores": p_scores,
            "labels": p_labels
        })

    metric.update(preds, targets)
    result = metric.compute()

    metrics = []
    metrics.append({
        "Subset": SUBSET,
        "mAP": result['map'].item(),
        "mAP_50": result['map_50'].item(),
        "mAP_75": result['map_75'].item(),
        "AP_Medium": result['map_medium'].item(),
        "AP_Large": result['map_large'].item()
    })

    df = pd.DataFrame(metrics)
    df.to_csv(f"{TEACHER_PATH}/SAM3_PigLife_performance.csv", index=False)

if __name__ == "__main__":
    evaluate_subset(SUBSET)