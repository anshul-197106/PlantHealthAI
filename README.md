# PlantHealth AI

A student project prototype for plant leaf disease screening, a rough visual severity estimate, and a severity-guided **sprinkler status simulation**.

## What it does

- Classifies a leaf image using a MobileNetV2 transfer-learning model.
- Shows an experimental color-based estimate of possible affected area.
- Displays a simulated sprinkler status for demonstration. No pump/valve hardware is controlled and no pesticide dose is calculated.
- Lets you explore the dashboard with manual demo inputs before a model has been trained.

## Important limitations

PlantVillage is a controlled image dataset. Good test accuracy there does not guarantee reliable field performance. The color proxy is not a validated infection measurement. Use an agricultural expert and label-approved instructions for real treatment decisions. Do not connect the demo output to real spraying equipment.

## Setup (Windows PowerShell)

Use Python 3.10–3.12 (Python 3.11 is a good choice). TensorFlow may not support the newest Python release immediately.

```powershell
cd "path\to\PlantHealthAI"
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the dashboard at any time, including before model training:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Get the data and train

1. The GitHub ZIP may contain manifests rather than the image files. Use the included downloader to fetch the image archive (about 2.2 GB) from Hugging Face and extract its color images into the expected folders:

   ```powershell
   .\.venv\Scripts\python.exe download_dataset.py
   ```

2. The expected layout is `data/raw/PlantVillage/`, with one folder per class. Example:

   ```text
   data/raw/PlantVillage/
     Tomato___Early_blight/
       image1.jpg
     Tomato___healthy/
       image2.jpg
   ```

3. Prepare reproducible train/validation/test folders:

   ```powershell
   .\.venv\Scripts\python.exe prepare_dataset.py
   ```

4. Train the model (the first run downloads pretrained MobileNetV2 weights):

   ```powershell
   .\.venv\Scripts\python.exe train.py
   ```

5. Evaluate on the held-out test split:

   ```powershell
   .\.venv\Scripts\python.exe evaluate.py
   ```

6. Run the app and upload a leaf photo:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

## GitHub and deployment

The repository should include the source files, `.streamlit/config.toml`, `models/best_model.keras`, `models/classes.json`, and the files under `reports/`. The `.gitignore` excludes the downloaded and processed image datasets, the virtual environment, and the extra training checkpoint. To host the app, connect the GitHub repository to Streamlit Community Cloud and choose `app.py` as the entry point.

The dataset split is random by image, so near-duplicate leaves may occur across splits. For a stronger project evaluation, collect independent field photos, report class-wise precision/recall and the confusion matrix, and describe failure cases.

## Project layout

```text
app.py                 Streamlit dashboard
prepare_dataset.py     Class-wise train/validation/test split
train.py               MobileNetV2 transfer-learning training
evaluate.py            Test report and confusion matrix
severity.py            Experimental color proxy
spray_logic.py         Display-only decision simulation
```

## Severity validation

To call severity validated, create manually annotated leaf and lesion masks for a representative held-out set. Compare estimated affected area with lesion pixels divided by leaf pixels, and report mean absolute error, sample count, and failure cases. Until then, show it only as an experimental proxy.
