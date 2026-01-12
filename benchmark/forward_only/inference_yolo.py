from ultralytics import YOLO
from transformers import Sam3Model, Sam3Processor
import os
import torch
import numpy as np
from PIL import Image
from itertools import product
import argparse

BASE_PATH = "/hd2/marcos/research/repos/pig-segmentation-distill/student/sizes/results/"
SOURCE_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife"

def benchmark_model(model_fn, repetitions=300, warmup=20):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions,))

    with torch.no_grad():
        for _ in range(warmup):
            _ = model_fn()
        torch.cuda.synchronize()
        for rep in range(repetitions):
            starter.record()
            _ = model_fn()
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    
    return np.mean(timings), np.std(timings)

def benchmark_yolo(imgsz=640, repetitions=300, warmup=20, **kwargs):
    experiment = kwargs["experiment"]
    model_name = kwargs["model_name"]
    model_path = os.path.join(BASE_PATH, experiment, model_name, "weights", "best.pt")
    yolo = YOLO(model_path)
    model = yolo.model.to("cuda").eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz).to("cuda")

    mean_ms, std_ms = benchmark_model(lambda: model(dummy_input), repetitions, warmup)
    return mean_ms, std_ms

def benchmark_sam(imgsz=1008, repetitions=300, warmup=20, **kwargs):
    model = Sam3Model.from_pretrained("facebook/sam3").to("cuda")
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    dummy_input = Image.fromarray(np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8))
    inputs = processor(images=dummy_input, text="pig", return_tensors="pt").to("cuda") # prompt aqui não é tão relevante

    mean_ms, std_ms = benchmark_model(lambda: model(**inputs), repetitions, warmup)
    return mean_ms, std_ms

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-r", "--repetitions", type=int, default=300, help="Number of repetitions for benchmarking")
    argparser.add_argument("-w", "--warmup", type=int, default=20, help="Number of warmup iterations before benchmarking")
    argparser.add_argument("-m", "--model", type=str, choices=["yolov8n", "yolov8s", "yolov8m", "sam3"], default="yolov8n", help="Model to benchmark")
    argparser.add_argument("-e", "--experiment", type=str, choices=["baseline", "sam3trained"], default="baseline", help="Experiment section to benchmark (only for YOLO models)")
    args = argparser.parse_args()

    if args.model == "sam3":
        mean_ms, std_ms = benchmark_sam(repetitions=args.repetitions, warmup=args.warmup)
        print(f"[SAM3] Avg Inference Time: {mean_ms:.2f} ms ± {std_ms:.2f} ms")
    else:
        mean_ms, std_ms = benchmark_yolo(experiment=args.experiment, model_name=args.model, repetitions=args.repetitions, warmup=args.warmup)
        print(f"[{args.experiment} | {args.model}] Avg Inference Time: {mean_ms:.2f} ms ± {std_ms:.2f} ms")