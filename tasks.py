from pathlib import Path
from dotenv import load_dotenv
from invoke import task
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import os
import shutil

console = Console()

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets"
PIGLIFE_DIR = DATASET_DIR / "piglife"
ZIP_DIR = PIGLIFE_DIR / "zip"
RAW_DIR = PIGLIFE_DIR / "raw"

load_dotenv()
PIGLIFE_URL = os.getenv("PIGLIFE_URL")

def ensure_directories():
    """Ensures the required directory structure exists."""
    for d in [ZIP_DIR, RAW_DIR, PIGLIFE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

@task
def download(c):
    """Downloads the PigLife dataset zip file."""
    ensure_directories()
    
    if not PIGLIFE_URL or PIGLIFE_URL == "YOUR_LINK_HERE":
        console.print("[bold red]Error:[/bold red] PIGLIFE_URL not defined in .env")
        raise ValueError("Please set PIGLIFE_URL in your .env file.")

    zip_file = ZIP_DIR / "piglife.zip"
    if not zip_file.exists():
        console.print(Panel(f"[bold #ff5f03]Starting dataset download...[/]\n[dim]{PIGLIFE_URL}[/dim]", title="Download", border_style="#13294c"))
        c.run(
            f'wget -q --show-progress --progress=bar:force:noscroll -O "{zip_file}" "{PIGLIFE_URL}"',
            pty=True,
        )
    else:
        console.print("[green]✔[/green] Dataset zip already exists. Skipping download.")

@task(pre=[download])
def setup(c):
    """Unzips, sanitizes, and converts the dataset to YOLO format."""
    ensure_directories()
    piglife_zip = ZIP_DIR / "piglife.zip"
    images_dir = RAW_DIR / "Image"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        
        if not any(RAW_DIR.iterdir()):
            progress.add_task(description="Extracting main dataset...", total=None)
            c.run(f'unzip -q "{piglife_zip}" -d "{RAW_DIR}"')

        images_dir.mkdir(parents=True, exist_ok=True)
        for zname in ("train.zip", "test.zip"):
            zip_path = images_dir / zname
            target_folder = images_dir / zname.replace(".zip", "")
            
            if zip_path.exists() and not target_folder.exists():
                progress.add_task(description=f"Unzipping {zname}...", total=None)
                c.run(f'unzip -q "{zip_path}" -d "{images_dir}"')

        c.run(f'unzip -q "{RAW_DIR / "Names.zip"}" -d "{RAW_DIR}"')

    console.print("[white]Data processing pipeline:[/white]")
    
    steps = [
        ("Sanitizing filenames", "sanitize.py"),
        ("Splitting dataset (train/val)", "split.py"),
        ("Converting COCO to YOLO format", "convert.py"),
        ("Organizing final directory structure", "organize.py"),
    ]

    for desc, script in steps:
        with console.status(f"[bold white]{desc}...[/bold white]"):
            c.run(f'python "{PROJECT_DIR}/datasets/{script}"', hide=True)
            
            if script == "split.py":
                anno_path = PIGLIFE_DIR / "coco" / "annotations"
                anno_path.mkdir(parents=True, exist_ok=True)
                
                source_json = images_dir / "pig_coco_test.json"
                dest_json = anno_path / "instances_test.json"
                
                if source_json.exists():
                    shutil.copy2(source_json, dest_json)
            
            console.print(f"  [green]✔[/green] {desc} completed.")

    macosx_dir = images_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

@task(pre=[setup])
def build(c):
    """Final task to signal completion."""
    console.print("[white]YOLO structure generated at:[/white] datasets/piglife/yolo")