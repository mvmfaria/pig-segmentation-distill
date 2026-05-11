from pathlib import Path
from env import Env
from invoke import task

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets"
PIGLIFE_DIR = DATASET_DIR / "piglife"
ZIP_DIR = PIGLIFE_DIR / "zip"
RAW_DIR = PIGLIFE_DIR / "raw"
PIGLIFE_URL = Env.get("PIGLIFE_URL")


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

    # c.run(f'wget -O "{ZIP_DIR / "piglife.zip"}" "{PIGLIFE_URL}"')

@task(pre=[download])
def setup(c):
    ensure_directories()
    c.run(f'unzip -q "{ZIP_DIR / "piglife.zip"}" -d "{RAW_DIR}"')

@task(pre=[setup])
def build(c):
    print("Finished! PigLife dataset is ready in the 'raw' directory.")