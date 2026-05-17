import os
import shutil

def sync_images_with_labels(label_dir, src_img_dir, dest_img_dir):
    os.makedirs(dest_img_dir, exist_ok=True)
    
    label_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f"Found {len(label_files)} labels in {label_dir}")
    
    for name in label_files:
        img_name = f"{name}.jpg" 
        src_path = os.path.join(src_img_dir, img_name)
        dest_path = os.path.join(dest_img_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
    
sync_images_with_labels(
    label_dir="datasets/piglife/yolo/human/labels/train", 
    src_img_dir="datasets/piglife/raw/Image/train",
    dest_img_dir="datasets/piglife/yolo/human/images/train"
)

sync_images_with_labels(
    label_dir="datasets/piglife/yolo/human/labels/val", 
    src_img_dir="datasets/piglife/raw/Image/train",
    dest_img_dir="datasets/piglife/yolo/human/images/val"
)

sync_images_with_labels(
    label_dir="datasets/piglife/yolo/human/labels/test", 
    src_img_dir="datasets/piglife/raw/Image/test",
    dest_img_dir="datasets/piglife/yolo/human/images/test"
)