from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data" / "processed" / "test"
MODEL_PATH = ROOT / "models" / "best_model.keras"
CLASS_PATH = ROOT / "models" / "classes.json"
REPORT_DIR = ROOT / "reports"


def main():
    for path in (TEST_DIR, MODEL_PATH, CLASS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Prepare data and train the model first.")
    classes = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, image_size=(224, 224), batch_size=32, shuffle=False, label_mode="int")
    if ds.class_names != classes:
        raise ValueError("Test class order differs from the saved class list.")
    model = tf.keras.models.load_model(MODEL_PATH)
    actual = np.concatenate([y.numpy() for _, y in ds])
    predicted = np.argmax(model.predict(ds, verbose=1), axis=1)
    report = classification_report(actual, predicted, target_names=classes, zero_division=0)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 0.45), max(7, len(classes) * 0.4)))
    ConfusionMatrixDisplay(confusion_matrix(actual, predicted), display_labels=classes).plot(ax=ax, xticks_rotation=90, colorbar=False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(report)
    print(f"Evaluation files saved in {REPORT_DIR}")


if __name__ == "__main__":
    main()
