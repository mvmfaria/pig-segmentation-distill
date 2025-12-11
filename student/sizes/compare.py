from ultralytics import YOLO
import os
import pandas as pd
import torch
import utils

PROJECT_NAME = "results"
DATA_CONFIG = "/hd2/marcos/research/repos/pig-segmentation-distill/student/data.yaml"
MODELS_FOLDER = "/hd2/marcos/research/repos/pig-segmentation-distill/models"
MODELS = ['yolo11n-seg.pt', 'yolo11s-seg.pt', 'yolo11m-seg.pt'] 
EPOCHS = 100
IMGSZ = 640
BATCH = 16

def run():
    results = []

    for model_name in MODELS:
        run_name = model_name.replace('.pt', '') 

        model = YOLO(os.path.join(MODELS_FOLDER, model_name))

        train_metrics = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            project=PROJECT_NAME,
            name=run_name,
            exist_ok=True,
            verbose=True
        )

        map50_mask, map50_95_mask, latency_ms, fps = utils.evaluate(model, train_metrics)

        results.append({
            "Model": model_name,
            "Parameters (M)": model.info()[1] / 1e6,
            "mAP50 (Mask)": map50_mask,
            "mAP50-95 (Mask)": map50_95_mask,
            "Latency (ms)": latency_ms,
            "FPS": fps
        })

        df = pd.DataFrame(results)
        df.to_csv(f"{PROJECT_NAME}/comparison.csv", index=False)

        del model
        torch.cuda.empty_cache()

        best_weights_path = os.path.join(PROJECT_NAME, run_name, 'weights', 'best.pt')
        
        val_model = YOLO(best_weights_path)
            
        val_model.val(
            data=DATA_CONFIG,
            split="test",
            project=PROJECT_NAME,
            name=f"{run_name}_test_eval"
        )
        del val_model

if __name__ == "__main__":
    run()