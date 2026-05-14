import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ANNOTATIONS_DIR = BASE_DIR / "piglife" / "raw" / "Image"

def sanitize_file_names(json_name):
    json_path = ANNOTATIONS_DIR / json_name

    with open(json_path, 'r') as f:
        data = json.load(f)

    for img in data['images']:
        img['file_name'] = Path(img['file_name']).name

    with open(json_path, 'w') as f:
        json.dump(data, f)
    
sanitize_file_names("pig_coco_test.json")
sanitize_file_names("pig_coco_train.json")