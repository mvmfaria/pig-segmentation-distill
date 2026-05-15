import json
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_ANNOTATIONS_DIR = BASE_DIR / "piglife" / "raw" / "Image"
OUTPUT_ANNOTATIONS_DIR = BASE_DIR / "piglife" / "coco" / "annotations"

def run_split(input_json, train_json, val_json, ratio=0.8):
    coco = COCO(input_json)
    img_ids = coco.getImgIds()

    train_ids, val_ids = train_test_split(img_ids, train_size=ratio, random_state=42)

    Path(train_json).parent.mkdir(parents=True, exist_ok=True)
    Path(val_json).parent.mkdir(parents=True, exist_ok=True)

    def save_subset(ids, output_path):
        res = {
            "images": coco.loadImgs(ids),
            "annotations": coco.loadAnns(coco.getAnnIds(imgIds=ids)),
            "categories": coco.loadCats(coco.getCatIds())
        }
        with open(output_path, 'w') as f:
            json.dump(res, f)

    save_subset(train_ids, train_json)
    save_subset(val_ids, val_json)

if __name__ == "__main__":
    run_split(INPUT_ANNOTATIONS_DIR / "pig_coco_train.json", OUTPUT_ANNOTATIONS_DIR / "instances_train.json", OUTPUT_ANNOTATIONS_DIR / "instances_val.json")