from ultralytics import YOLO
import os
import torch
import numpy as np
import argparse

BASE_PATH = "/hd2/marcos/research/repos/pig-segmentation-distill/student/sizes/results/"
SOURCE_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife"

def benchmark_yolo(subset_name, experiment, model_name, warmup=20):
    model_path = os.path.join(BASE_PATH, experiment, model_name, "weights", "best.pt")
    model = YOLO(model_path).eval()

    image_dir = os.path.join(SOURCE_ROOT, subset_name, "images")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    timings = []
    with torch.no_grad():
        for _ in range(warmup):
            model.predict(
                os.path.join(image_dir, image_files[0]),
                imgsz=640,
                verbose=False,
            )
        torch.cuda.synchronize()
        for img_name in image_files:
            r = model.predict(
                os.path.join(image_dir, img_name),
                imgsz=640,
                verbose=False,
            )
            timings.append(sum(r[0].speed.values()))
            torch.cuda.synchronize()
    
    return np.mean(timings), np.std(timings)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-w", "--warmup", type=int, default=20, help="Number of warmup iterations before benchmarking")
    argparser.add_argument("-m", "--model", type=str, choices=["yolov8n", "yolov8s", "yolov8m"], default="yolov8n", help="Model to benchmark")
    argparser.add_argument("-e", "--experiment", type=str, choices=["baseline", "sam3trained"], default="baseline", help="Experiment section to benchmark (only for YOLO models)")
    args = argparser.parse_args()

    mean_ms, std_ms = benchmark_yolo("test", experiment=args.experiment, model_name=args.model, warmup=args.warmup)
    print(f"[{args.experiment} | {args.model}] Avg Inference Time: {mean_ms:.2f} ms ± {std_ms:.2f} ms")
