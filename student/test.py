from ultralytics import YOLO

def run_inference():
    model = YOLO('/hd2/marcos/research/repos/cross-domain-pig-detection/runs/segment/train2/weights/best.pt')

    results = model.predict(
        source='/hd2/marcos/research/repos/pig-segmentation-distill/videos/1020s1120s2003-2s5301-1.mp4', 
        save=True,
        conf=0.5,
        imgsz=640,
        project='/hd2/marcos/research/repos/pig-segmentation-distill/videos/inferences',
        stream=True
    )

    for _ in results:
        pass

if __name__ == '__main__':
    run_inference()