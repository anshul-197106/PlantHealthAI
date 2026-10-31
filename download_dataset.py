"""Download PlantVillage images from Hugging Face into data/raw/PlantVillage."""
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "raw" / "PlantVillage"
DATASET_NAME = "mohanty/PlantVillage"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading the PlantVillage archive (about 2.2 GB). Make sure you have enough free disk space.")
    archive_path = hf_hub_download(repo_id=DATASET_NAME, filename="data.zip", repo_type="dataset")
    print("Extracting color leaf photos into class folders. This may take several minutes...")
    saved = 0
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    with ZipFile(archive_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        for member in members:
            parts = Path(member.filename.replace("\\", "/")).parts
            lowered = [part.lower() for part in parts]
            if "color" not in lowered:
                continue
            color_index = lowered.index("color")
            if len(parts) <= color_index + 2 or Path(parts[-1]).suffix.lower() not in image_exts:
                continue
            class_name = parts[color_index + 1]
            class_dir = OUT_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            target = class_dir / parts[-1]
            with archive.open(member) as source, target.open("wb") as destination:
                destination.write(source.read())
            saved += 1
            if saved % 1000 == 0:
                print(f"Saved {saved} images...")
    if saved == 0:
        raise RuntimeError("No color images found in the downloaded archive. Check the archive contents.")
    print(f"Finished: extracted {saved} images in {OUT_DIR}")
    print("Now run: .\\.venv\\Scripts\\python.exe prepare_dataset.py")


if __name__ == "__main__":
    main()
