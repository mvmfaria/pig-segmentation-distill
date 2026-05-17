from ultralytics.data.converter import convert_coco
import argparse

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--source",
        type=str,
        required=True,
    )
    
    args = parser.parse_args()
    
    convert_coco(
        labels_dir=f"datasets/piglife/coco/{args.source}/annotations/", 
        save_dir=f"datasets/piglife/yolo/{args.source}/",
        use_segments=False,
        cls91to80=False
    )

if __name__ == "__main__":
    main()