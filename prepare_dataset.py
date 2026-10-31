from pathlib import Path
import random
import shutil

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "PlantVillage"
OUT_DIR = ROOT / "data" / "processed"
SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {RAW_DIR}\nCreate it and put one image folder per class inside.")
    classes = {}
    for folder in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(files) >= 10:
            classes[folder.name] = files
        elif files:
            print(f"Skipping {folder.name}: need at least 10 images; found {len(files)}")
    if len(classes) < 2:
        raise RuntimeError("Need at least two classes with 10 or more images each.")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    random.seed(SEED)
    for label, files in classes.items():
        train, remainder = train_test_split(files, test_size=0.30, random_state=SEED)
        val, test = train_test_split(remainder, test_size=0.50, random_state=SEED)
        for split, paths in (("train", train), ("val", val), ("test", test)):
            target = OUT_DIR / split / label
            target.mkdir(parents=True, exist_ok=True)
            for i, source in enumerate(paths):
                shutil.copy2(source, target / f"{source.stem}_{i}{source.suffix.lower()}")
            print(f"{split:5} {label}: {len(paths)} images")
    print(f"\nDataset splits created at {OUT_DIR}")
    print("Note: random image splits may contain near-duplicates. Keep independent field images for a realistic evaluation.")


if __name__ == "__main__":
    main()
