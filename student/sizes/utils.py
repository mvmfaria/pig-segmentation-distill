import torch
import time
import torch

IMGSZ = 640

def measure_inference_time(model, warmups=10, runs=100, imgsz=640):
    if torch.cuda.is_available():
        model = model.to('cuda')
        
    device = next(model.parameters()).device
    
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)

    for _ in range(warmups):
        model(dummy_input) 

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(runs):
        model(dummy_input)
        
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = ((end - start) / runs) * 1000
    return avg_time_ms

def evaluate(model, train_metrics):
    map50_box = train_metrics.box.map50
    map50_95_box = train_metrics.box.map
    latency_ms = measure_inference_time(model.model, imgsz=IMGSZ)
    fps = 1000 / latency_ms

    return map50_box, map50_95_box, latency_ms, fps