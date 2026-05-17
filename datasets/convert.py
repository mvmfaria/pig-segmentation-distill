from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="datasets/piglife/coco/human/annotations/", 
    save_dir="datasets/piglife/yolo/human/", 
    use_segments=False, 
    cls91to80=False
)