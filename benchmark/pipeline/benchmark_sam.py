from transformers import Sam3Model, Sam3Processor
from PIL import Image
import os
import torch
import numpy as np
import argparse
import time

SOURCE_ROOT = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife"

def benchmark_sam(subset_name, warmup=20):
    model = Sam3Model.from_pretrained(
    "facebook/sam3",
    torch_dtype=torch.bfloat16
    ).to("cuda")
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    image_dir = os.path.join(SOURCE_ROOT, subset_name, "images")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    timings = []
    with torch.no_grad():
        for _ in range(warmup):
            img = Image.open(os.path.join(image_dir, image_files[0])).convert("RGB")
            inputs = processor(images=img, text="pig", return_tensors="pt").to("cuda", dtype=torch.bfloat16)
            outputs = model(**inputs)
            _ = processor.post_process_instance_segmentation(outputs, target_sizes=inputs.get("original_sizes").tolist())
        torch.cuda.synchronize()
        for img_name in image_files:
            img = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
            start_time = time.perf_counter()
            inputs = processor(images=img, text="pig", return_tensors="pt").to("cuda", dtype=torch.bfloat16)
            outputs = model(**inputs)
            _ = processor.post_process_instance_segmentation(outputs, target_sizes=inputs.get("original_sizes").tolist())
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)
    
    return np.mean(timings), np.std(timings)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-w", "--warmup", type=int, default=20, help="Number of warmup iterations before benchmarking")
    args = argparser.parse_args()

    mean_ms, std_ms = benchmark_sam("test", warmup=args.warmup)
    print(f"[SAM3] Avg Inference Time: {mean_ms:.2f} ms ± {std_ms:.2f} ms")
