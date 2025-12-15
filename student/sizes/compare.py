from ultralytics import YOLO
import os
import pandas as pd
import torch
import utils

PROJECT_NAME = "results"
CONTEXT = "baseline"
DATA_CONFIG = f"/hd2/marcos/research/repos/pig-segmentation-distill/student/sizes/results/{CONTEXT}/data.yaml"
GT_ROOT_TEST = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife/test/images"
MODELS_FOLDER = f"/hd2/marcos/research/repos/pig-segmentation-distill/models/detection"
MODELS = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']

EPOCHS = 100
IMGSZ = 640
BATCH = 4
WORKERS = 2

def save_test_predictions(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    results = model.predict(
        source=GT_ROOT_TEST, 
        stream=True, 
        conf=0.001, 
        iou=0.7, 
        save=False, 
        verbose=False
    )
    
    for result in results:
        txt_name = os.path.splitext(os.path.basename(result.path))[0] + ".txt"
        save_path = os.path.join(output_dir, txt_name)
        
        lines = []
        boxes = result.boxes
        for i in range(len(boxes)):
            xywh = boxes.xywhn[i].cpu().tolist()
            cls = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            
            lines.append(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f} {conf:.6f}")
            
        with open(save_path, "w") as f:
            f.write("\n".join(lines))

def run():
    os.makedirs(f"{PROJECT_NAME}/{CONTEXT}", exist_ok=True)
    results = []

    for model_name in MODELS:
        run_name = model_name.replace('.pt', '')

        model_path = os.path.join(MODELS_FOLDER, model_name)

        model = YOLO(model_path)

        train_metrics = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            workers=WORKERS,
            project=f"{PROJECT_NAME}/{CONTEXT}",
            name=run_name,
            exist_ok=True,
            verbose=True
        )

        map50_box, map50_95_box, latency_ms, fps = utils.evaluate(model, train_metrics)

        results.append({
            "Model": model_name,
            "Parameters (M)": model.info()[1] / 1e6,
            "mAP50 (Box)": map50_box,
            "mAP50-95 (Box)": map50_95_box,
            "Latency (ms)": latency_ms,
            "FPS": fps
        })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{PROJECT_NAME}/{CONTEXT}/comparison.csv", index=False)

        del model
        torch.cuda.empty_cache()

        best_weights_path = os.path.join(PROJECT_NAME, CONTEXT, run_name, 'weights', 'best.pt')
        
        best_model = YOLO(best_weights_path)
            
        pred_dir = os.path.join(PROJECT_NAME, CONTEXT, run_name, "test_predictions_txt")
        save_test_predictions(best_model, pred_dir)
        
        del best_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run()