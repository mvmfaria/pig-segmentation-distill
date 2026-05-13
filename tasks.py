from pathlib import Path
from dotenv import load_dotenv
from invoke import task
import os

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets"
PIGLIFE_DIR = DATASET_DIR / "piglife"
ZIP_DIR = PIGLIFE_DIR / "zip"
RAW_DIR = PIGLIFE_DIR / "raw"
load_dotenv()
PIGLIFE_URL = os.getenv("PIGLIFE_URL")

def ensure_directories():
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PIGLIFE_DIR.mkdir(parents=True, exist_ok=True)

@task
def download(c):
    ensure_directories()

    if not PIGLIFE_URL or PIGLIFE_URL == "COLE_SEU_LINK_AQUI":
        raise ValueError(
            "Define PIGLIFE_URL with the link for the PigLife dataset before running this task."
        )

    if not (ZIP_DIR / "piglife.zip").exists():
        c.run(f'wget -O "{ZIP_DIR / "piglife.zip"}" "{PIGLIFE_URL}"')

@task(pre=[download])
def setup(c):
    ensure_directories()

    piglife_zip = ZIP_DIR / "piglife.zip"
    if not os.listdir(RAW_DIR):
        c.run(f'unzip -q "{piglife_zip}" -d "{RAW_DIR}"')

    images_dir = RAW_DIR / "Image"
    images_dir.mkdir(parents=True, exist_ok=True)

    for zname in ("train.zip", "test.zip"):
        zip_path = images_dir / zname
        if zip_path.exists():
            if not (images_dir / zname.replace(".zip", "")).exists():
                c.run(f'unzip -q "{zip_path}" -d "{images_dir}"')
    
    c.run(f'rm -rf {images_dir}/__MACOSX')

    if not (PIGLIFE_DIR / "coco" / "annotations").exists():
        c.run(f'mkdir -p {PIGLIFE_DIR / "coco" / "annotations"}')
        c.run(f'mv {images_dir / "pig_coco_test.json"} {PIGLIFE_DIR / "coco" / "annotations" / "instances_test.json"}')
        c.run(f'mv {images_dir / "pig_coco_train.json"} {PIGLIFE_DIR / "coco" / "annotations" / "instances_train.json"}')
        c.run(f'python {PROJECT_DIR / "datasets" / "sanitize.py"}')
    
    c.run(f'python {PROJECT_DIR / "datasets" / "convert.py"}')

@task(pre=[setup])
def build(c):
    print("Finished! PigLife dataset is ready in the 'raw' directory.")