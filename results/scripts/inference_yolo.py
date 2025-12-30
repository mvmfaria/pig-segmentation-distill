from ultralytics import YOLO
from transformers import Sam3Model, Sam3Processor
import torch
import numpy as np

def benchmark_model(model, dummy_input, repetitions=300, warmup=20):
    model.eval()
    
    # Ensure dummy_input is a tuple for unpacking, even if it's a single tensor
    if not isinstance(dummy_input, (list, tuple)):
        inputs = (dummy_input,)
    else:
        inputs = dummy_input

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs) # Unpacks the tuple into x and boxes
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(*inputs) # Unpacks the tuple here as well
            ender.record()
            
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            
    mean_ms = np.sum(timings) / repetitions
    std_ms = np.std(timings)
    return mean_ms, std_ms

# yolo_wrapper = YOLO("/hd2/marcos/research/repos/pig-segmentation-distill/student/sizes/results/sam3trained/yolov8s/weights/best.pt")
# model = yolo_wrapper.model.to("cuda")

# dummy_input = torch.randn(1, 3, 640, 640).to("cuda")

# mean_yolo, std_yolo = benchmark_model(model, dummy_input)
# print(f"YOLOv8 Inference: {mean_yolo:.2f} ms ± {std_yolo:.2f}")

model = Sam3Model.from_pretrained("facebook/sam3").to("cuda")
# dummy_input for SAM3 is more complex (requires pixel_values and input_boxes/text)
# We simulate a typical 1024x1024 input
dummy_pixel_values = torch.randn(1, 3, 1024, 1024).to("cuda")
dummy_boxes = torch.tensor([[[100, 100, 200, 200]]]).to("cuda") # [batch, num_boxes, 4]

# Define a tiny wrapper so the benchmark function can call it simply
class SAM3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, boxes):
        return self.model(pixel_values=x, input_boxes=boxes)

sam_bench_model = SAM3Wrapper(model)
mean_sam, std_sam = benchmark_model(sam_bench_model, (dummy_pixel_values, dummy_boxes),  repetitions=100) # SAM is slower, 100 is enough
print(f"SAM3 Inference: {mean_sam:.2f} ms ± {std_sam:.2f}")