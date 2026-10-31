from datetime import datetime
from pathlib import Path
import json

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from severity import estimate_visual_severity
from spray_logic import simulated_spraying_decision


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.keras"
CLASS_PATH = ROOT / "models" / "classes.json"
REPORT_PATH = ROOT / "reports" / "classification_report.txt"
CONFUSION_PATH = ROOT / "reports" / "confusion_matrix.png"

st.set_page_config(
    page_title="PlantHealth AI | Crop intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --forest: #103b2b; --forest-2: #174b36; --emerald: #16845b;
      --mint: #e7f4ed; --ink: #18251f; --muted: #708078;
      --paper: #f4f7f4; --line: #e2e9e3; --white: #ffffff;
    }
    html, body, [class*="css"] { font-family: 'Aptos', 'Segoe UI', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244,247,244,.92); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #103b2b 0%, #0d3023 100%); }
    [data-testid="stSidebar"] * { color: #eef8f1; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .55rem .75rem; border-radius: 10px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.10); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c3d9ca; }
    .block-container { max-width: 1440px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Aptos Display', 'Segoe UI', sans-serif !important; color: var(--ink); letter-spacing: -.035em; }
    h1 { font-size: clamp(2rem, 3vw, 3.1rem) !important; }
    h2 { font-size: 1.6rem !important; }
    [data-testid="stMetric"] { background: var(--white); border: 1px solid var(--line); padding: 1rem 1.1rem; border-radius: 16px; box-shadow: 0 5px 18px rgba(20,55,38,.035); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--forest); font-family: 'Aptos Display', 'Segoe UI', sans-serif; }
    [data-testid="stFileUploader"] section { border: 1.5px dashed #86b39a; background: #f8fcf9; border-radius: 16px; }
    [data-testid="stFileUploader"] section:hover { border-color: var(--emerald); background: #f0f8f3; }
    [data-testid="stButton"] button { border-radius: 11px; font-weight: 700; border: 0; transition: transform .16s ease, box-shadow .16s ease; }
    [data-testid="stButton"] button[kind="primary"] { background: var(--emerald); color: white; }
    [data-testid="stButton"] button:hover { transform: translateY(-1px); box-shadow: 0 8px 18px rgba(22,132,91,.18); }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 14px; background: white; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    [data-testid="stProgress"] > div > div { background: var(--emerald); }
    .brand-lockup { display:flex; align-items:center; gap:.75rem; padding:.4rem .2rem 1.2rem; }
    .brand-mark { width:42px; height:42px; border-radius:13px; display:grid; place-items:center; font-size:24px; background:#d9f0e2; }
    .brand-name { font-family:'Aptos Display','Segoe UI',sans-serif; font-size:1.08rem; font-weight:800; line-height:1.15; color:white; }
    .brand-sub { font-size:.72rem; color:#b8d3c2; margin-top:3px; }
    .hero { background: radial-gradient(circle at 85% 15%, rgba(103,196,142,.36), transparent 27%), linear-gradient(120deg,#103b2b,#19573c); color:white; padding:2rem 2.2rem; border-radius:22px; margin-bottom:1.35rem; box-shadow:0 14px 35px rgba(16,59,43,.14); }
    .hero h1,.hero h2,.hero p { color:white !important; margin-top:0; }
    .hero p { color:#d9ebe0 !important; max-width:740px; margin-bottom:.2rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; color:#a7dbba; margin-bottom:.55rem; }
    .section-kicker { font-size:.76rem; color:var(--emerald); text-transform:uppercase; letter-spacing:.12em; font-weight:700; margin-bottom:.3rem; }
    .surface { background:white; border:1px solid var(--line); border-radius:17px; padding:1.1rem 1.25rem; box-shadow:0 6px 18px rgba(20,55,38,.035); transition:transform .18s ease, box-shadow .18s ease; }
    .surface:hover { transform:translateY(-2px); box-shadow:0 12px 26px rgba(20,55,38,.08); }
    .surface-title { font-family:'Aptos Display','Segoe UI',sans-serif; color:var(--forest); font-size:1.05rem; font-weight:800; margin-bottom:.55rem; }
    .muted { color:var(--muted); font-size:.9rem; }
    .status-pill { display:inline-block; padding:.35rem .7rem; border-radius:999px; background:#e7f4ed; color:#12613e; font-weight:700; font-size:.78rem; }
    .result-title { font-family:'Aptos Display','Segoe UI',sans-serif; color:var(--forest); font-size:1.45rem; font-weight:800; line-height:1.2; }
    .flow { display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; margin:.5rem 0 1rem; }
    .flow-step { padding:.55rem .75rem; border-radius:10px; background:#eef3ef; color:#6a7c71; font-size:.78rem; font-weight:700; }
    .flow-step.done { background:#def1e5; color:#14603e; }
    .flow-arrow { color:#9caea2; }
    .small-note { border-left:3px solid #d99b35; background:#fff8e8; padding:.75rem .9rem; border-radius:0 10px 10px 0; color:#76571e; font-size:.84rem; }
    .footer-note { color:var(--muted); font-size:.78rem; padding-top:1rem; border-top:1px solid var(--line); margin-top:2rem; }
    @media (max-width: 720px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero { padding:1.4rem; border-radius:16px; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


@st.cache_data
def read_classes(path: str, modified: float):
    del modified  # Include mtime in the cache key so retraining refreshes labels.
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def get_classes():
    if not CLASS_PATH.exists():
        return []
    return read_classes(str(CLASS_PATH), CLASS_PATH.stat().st_mtime)


def read_evaluation_report(path: Path):
    """Parse the sklearn classification report generated by evaluate.py."""
    if not path.exists():
        return None
    rows, summary = [], {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("accuracy"):
            values = text.split()
            if len(values) >= 3:
                try:
                    summary["accuracy"] = float(values[1])
                    summary["support"] = int(values[2])
                except ValueError:
                    pass
            continue
        columns = text.rsplit(maxsplit=4)
        if len(columns) != 5:
            continue
        name, precision, recall, f1, support = columns
        try:
            row = {"Class": name, "Precision": float(precision), "Recall": float(recall), "F1 score": float(f1), "Images": int(support)}
        except ValueError:
            continue
        if name in {"macro avg", "weighted avg", "micro avg", "samples avg"}:
            summary[name] = row
        else:
            rows.append(row)
    if not rows or "accuracy" not in summary:
        return None
    return {"classes": rows, "summary": summary}


def get_report():
    report = read_evaluation_report(REPORT_PATH)
    if report and MODEL_PATH.exists() and REPORT_PATH.stat().st_mtime < MODEL_PATH.stat().st_mtime:
        report["stale"] = True
    elif report:
        report["stale"] = False
    return report


def friendly_label(label: str) -> str:
    return label.replace("___", " · ").replace("_", " ")


def display_class_name(label: str) -> tuple[str, str]:
    if "___" in label:
        crop, condition = label.split("___", 1)
        return crop.replace("_", " "), condition.replace("_", " ")
    return "Plant", label.replace("_", " ")


def init_state():
    st.session_state.setdefault("analysis_history", [])
    st.session_state.setdefault("current_analysis", None)


def metric_or_dash(report, key, percent=False):
    if not report:
        return "—"
    value = report["summary"].get(key)
    if value is None:
        return "—"
    if isinstance(value, dict):
        value = value.get("F1 score")
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if percent else value


def render_sidebar(classes, report):
    with st.sidebar:
        st.markdown(
            '<div class="brand-lockup"><div class="brand-mark">🌿</div><div><div class="brand-name">PlantHealth AI</div><div class="brand-sub">Crop intelligence platform</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-kicker' style='color:#9fd3b2'>WORKSPACE</div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            ["Overview", "Disease Detection", "Performance", "Plant Library", "About Project"],
            label_visibility="collapsed",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        status = "Model ready" if MODEL_PATH.exists() and CLASS_PATH.exists() else "Model not trained"
        status_color = "#94e2b0" if status == "Model ready" else "#f3ce83"
        st.markdown(
            f"<div style='padding:.8rem .9rem;border:1px solid #ffffff26;border-radius:13px;background:#ffffff0d'><div style='font-size:.7rem;color:#b8d3c2;text-transform:uppercase;letter-spacing:.1em'>SYSTEM STATUS</div><div style='margin-top:.4rem;font-weight:700;color:{status_color}'>● {status}</div><div style='font-size:.76rem;color:#b8d3c2;margin-top:.3rem'>{len(classes)} supported classes</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.caption("Research prototype · decision support only")
        return page


def render_hero(eyebrow, title, subtitle):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_evaluation_metrics(report, columns=4):
    if not report:
        st.info("Model evaluation is not available yet. Run `python evaluate.py` after training.")
        return
    summary = report["summary"]
    macro, weighted = summary.get("macro avg", {}), summary.get("weighted avg", {})
    cards = st.columns(columns)
    metrics = [
        ("Test accuracy", metric_or_dash(report, "accuracy", True)),
        ("Macro precision", f'{macro.get("Precision", 0) * 100:.1f}%'),
        ("Macro recall", f'{macro.get("Recall", 0) * 100:.1f}%'),
        ("Macro F1", f'{macro.get("F1 score", 0) * 100:.1f}%'),
    ]
    if columns >= 5:
        metrics.append(("Images tested", f'{summary.get("support", 0):,}'))
    for card, (label, value) in zip(cards, metrics):
        card.metric(label, value)
    if report.get("stale"):
        st.warning("This evaluation report is older than the current model. Run `python evaluate.py` to refresh the metrics.")
    else:
        st.caption(f"Measured on {summary.get('support', 0):,} held-out PlantVillage images. Field performance may differ.")


def render_overview(classes, report):
    render_hero("PLANT HEALTH INTELLIGENCE", "A clearer view of crop health.", "Screen leaf images, review model evidence, and explore a safe sprinkler simulation in one workspace.")
    crop_names = sorted({display_class_name(label)[0] for label in classes})
    recent = st.session_state.analysis_history
    summary = report["summary"] if report else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Supported conditions", f"{len(classes):,}", help="Includes healthy and disease labels represented in the trained model.")
    c2.metric("Plant crops", f"{len(crop_names):,}")
    c3.metric("Test accuracy", metric_or_dash(report, "accuracy", True))
    c4.metric("Images evaluated", f"{summary.get('support', 0):,}" if report else "—")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("### Model performance")
        if report:
            render_evaluation_metrics(report, columns=4)
            weakest = sorted(report["classes"], key=lambda row: row["Recall"])[:5]
            if weakest:
                st.markdown("#### Classes to review")
                st.dataframe(weakest, hide_index=True, use_container_width=True)
        else:
            st.info("Run model evaluation to populate measured performance metrics.")
    with right:
        st.markdown("### Recent analysis")
        if recent:
            for item in recent[:5]:
                crop, condition = display_class_name(item["label"])
                st.markdown(
                    f'<div class="surface" style="margin-bottom:.55rem;padding:.8rem 1rem"><div style="display:flex;justify-content:space-between;gap:.5rem"><b style="color:#174b36">{crop} · {condition}</b><span class="status-pill">{item["confidence"]:.1f}%</span></div><div class="muted" style="margin-top:.25rem">{item["time"]}</div></div>',
                    unsafe_allow_html=True,
                )
            st.caption("History is held in this browser session and is not written to disk.")
        else:
            st.markdown('<div class="surface"><div class="surface-title">Your workspace is ready</div><div class="muted">Run your first leaf analysis to see recent activity here.</div></div>', unsafe_allow_html=True)

    st.markdown("<div class='small-note'>Research prototype: diagnoses can be wrong. The color estimate is experimental, and the sprinkler status is simulated only. Confirm treatment decisions with qualified agricultural guidance.</div>", unsafe_allow_html=True)


def render_detection(classes):
    render_hero("IMAGE SCREENING", "Analyze a leaf", "Upload a clear photo to view the model's leading class predictions and experimental visual estimate.")
    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        st.warning("A trained model was not found. Train the model before using image analysis.")
        st.markdown("### Sprinkler simulation demo")
        st.caption("Manual inputs demonstrate the interface only. No pump, valve, or pesticide quantity is connected to these controls.")
        condition = st.selectbox("Demo plant condition", ["Needs inspection", "Healthy", "Disease suspected"], key="fallback_condition")
        level = st.slider("Illustrative affected area (%)", 0, 100, 25, key="fallback_level")
        if st.button("Preview demo status", type="primary", key="fallback_preview"):
            label = {"Needs inspection": "uncertain", "Healthy": "healthy", "Disease suspected": "disease suspected"}[condition]
            confidence = 0.0 if condition == "Needs inspection" else 1.0
            decision = simulated_spraying_decision(label, confidence, {"value": level, "status": "manual_demo"})
            st.info(f"{decision['level']} · {decision['action']}")
            st.caption(decision["reason"])
        return

    upload_col, tip_col = st.columns([1.35, 1])
    with upload_col:
        uploaded = st.file_uploader("Drop a leaf image here or browse files", type=["jpg", "jpeg", "png"], key="leaf_upload")
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Ready to analyze", use_container_width=True)
                if st.button("Analyze plant", type="primary", use_container_width=True):
                    with st.spinner("Analyzing leaf image…"):
                        model = load_model(str(MODEL_PATH))
                        input_array = np.asarray(image.resize((224, 224)), dtype=np.float32)
                        batch = np.expand_dims(input_array, axis=0)
                        probabilities = model.predict(batch, verbose=0)[0]
                        order = np.argsort(probabilities)[::-1][:3]
                        predicted_index = int(order[0])
                        label = classes[predicted_index]
                        confidence = float(probabilities[predicted_index])
                        severity = estimate_visual_severity(image)
                        decision = simulated_spraying_decision(label, confidence, severity)
                        result = {
                            "label": label,
                            "confidence": confidence,
                            "top3": [{"label": classes[int(index)], "score": float(probabilities[int(index)])} for index in order],
                            "severity": severity,
                            "decision": decision,
                            "image": image.copy(),
                            "time": datetime.now().strftime("%d %b %Y · %I:%M %p"),
                        }
                        st.session_state.current_analysis = result
                        history = st.session_state.analysis_history
                        history.insert(0, {"label": label, "confidence": confidence * 100, "time": result["time"]})
                        st.session_state.analysis_history = history[:10]
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file could not be opened as a valid image. Please choose a JPG or PNG leaf photo.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
    with tip_col:
        st.markdown('<div class="surface"><div class="surface-title">For a clearer result</div><div class="muted">• Keep one leaf centered in the frame<br>• Use even natural light<br>• Avoid blur and heavy shadows<br>• Include the full leaf when possible</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="small-note">The model was trained on PlantVillage images. Field photos can look different from its training data.</div>', unsafe_allow_html=True)

    result = st.session_state.current_analysis
    if result:
        st.divider()
        st.markdown("### Analysis result")
        crop, condition = display_class_name(result["label"])
        a, b, c = st.columns([1.1, 1, 1])
        with a:
            st.markdown(f'<div class="surface"><div class="section-kicker">PREDICTED CONDITION</div><div class="result-title">{condition}</div><div class="muted" style="margin-top:.35rem">{crop}</div><div style="margin-top:.75rem"><span class="status-pill">{result["confidence"] * 100:.1f}% model confidence</span></div></div>', unsafe_allow_html=True)
        with b:
            st.markdown('<div class="surface"><div class="section-kicker">EXPERIMENTAL VISUAL ESTIMATE</div>', unsafe_allow_html=True)
            severity = result["severity"]
            if severity.get("value") is None:
                st.info(severity["message"])
            else:
                st.metric("Possible affected-color area", f'{severity["value"]:.1f}%')
                st.progress(float(np.clip(severity["value"] / 100.0, 0, 1)))
                st.caption("Color proxy only · not validated infection severity")
            st.markdown("</div>", unsafe_allow_html=True)
        with c:
            decision = result["decision"]
            st.markdown(f'<div class="surface"><div class="section-kicker">SIMULATED SPRINKLER STATUS</div><div class="result-title">{decision["level"]}</div><div class="muted" style="margin-top:.45rem">{decision["action"]}</div></div>', unsafe_allow_html=True)

        st.markdown("<div class='flow'><span class='flow-step done'>1 · Leaf image</span><span class='flow-arrow'>→</span><span class='flow-step done'>2 · AI screening</span><span class='flow-arrow'>→</span><span class='flow-step done'>3 · Visual proxy</span><span class='flow-arrow'>→</span><span class='flow-step done'>4 · Human review</span></div>", unsafe_allow_html=True)
        st.caption(decision["reason"])
        st.markdown("#### Top 3 predictions")
        for item in result["top3"]:
            st.write(f"**{friendly_label(item['label'])}** · {item['score'] * 100:.1f}%")
            st.progress(float(item["score"]))
        st.caption("Model confidence is not the probability that a field diagnosis is correct.")

    with st.expander("Sprinkler simulation demo controls"):
        st.caption("These manual inputs are illustrative only. No equipment is connected and no treatment quantity is calculated.")
        demo_condition = st.selectbox("Demo condition", ["Needs inspection", "Healthy", "Disease suspected"], key="demo_condition")
        demo_level = st.slider("Illustrative affected area (%)", 0, 100, 25, key="demo_level")
        if st.button("Preview demo status", key="demo_preview"):
            label = {"Needs inspection": "uncertain", "Healthy": "healthy", "Disease suspected": "disease suspected"}[demo_condition]
            confidence = 0.0 if demo_condition == "Needs inspection" else 1.0
            decision = simulated_spraying_decision(label, confidence, {"value": demo_level, "status": "manual_demo"})
            st.info(f"{decision['level']} · {decision['action']}")
            st.caption(decision["reason"])


def render_performance(report):
    render_hero("MODEL QUALITY", "Evidence behind every prediction.", "Review held-out test results, compare class-level metrics, and inspect where the model confuses conditions.")
    if not report:
        st.info("No evaluation report found. Run `python evaluate.py` after training to create the report and confusion matrix.")
        return
    render_evaluation_metrics(report, columns=5)
    if report.get("stale"):
        st.warning("Metrics shown here were generated before the current model file. Re-run evaluation for an up-to-date report.")
    tabs = st.tabs(["Class metrics", "Confusion matrix", "How to read this"])
    with tabs[0]:
        rows = sorted(report["classes"], key=lambda row: row["Recall"])
        st.markdown("### Per-class results")
        st.caption("Sorted by recall, lowest first, to make classes that need review easy to find.")
        st.dataframe(rows, hide_index=True, use_container_width=True)
    with tabs[1]:
        if CONFUSION_PATH.exists():
            st.image(str(CONFUSION_PATH), use_container_width=True)
            st.caption("Rows show true labels; columns show predicted labels. Dark diagonal cells indicate correct predictions.")
        else:
            st.info("Confusion matrix image not found. Run `python evaluate.py` to regenerate it.")
    with tabs[2]:
        st.markdown("- **Precision:** when the model predicts a class, how often that class is correct.\n- **Recall:** how many images of a class the model successfully finds.\n- **F1:** balance between precision and recall.\n- **Macro average:** treats every class equally.\n- **Weighted average:** gives larger classes more influence.")
        st.markdown('<div class="small-note">These metrics describe the held-out PlantVillage split. They do not establish performance on farm images, where lighting, backgrounds, and symptom stages may differ.</div>', unsafe_allow_html=True)


def render_library(classes):
    render_hero("CROP KNOWLEDGE", "Supported crops and conditions.", "Explore the class labels included in the currently trained model.")
    if not classes:
        st.info("Class labels are unavailable. Train the model to create `models/classes.json`.")
        return
    grouped = {}
    for label in classes:
        crop, condition = display_class_name(label)
        grouped.setdefault(crop, []).append(condition)
    st.markdown(f"**{len(grouped)} crops** · **{len(classes)} supported labels**")
    crop = st.selectbox("Choose a crop", sorted(grouped))
    st.markdown(f"### {crop}")
    cols = st.columns(3)
    for index, condition in enumerate(sorted(grouped[crop])):
        kind = "Healthy class" if "healthy" in condition.lower() else "Disease / condition class"
        cols[index % len(cols)].markdown(f'<div class="surface" style="margin-bottom:.7rem"><div class="surface-title">{condition}</div><span class="status-pill">{kind}</span></div>', unsafe_allow_html=True)
    with st.expander("Disease information"):
        selected = st.selectbox("Select a class label", sorted(grouped[crop]), key="library_condition")
        st.write(f"**{crop} · {selected}** is a label represented in the model training dataset.")
        st.caption("This project provides image classification, not a verified agronomic diagnosis or treatment recommendation. Confirm symptoms with an agricultural expert.")


def render_about():
    render_hero("ABOUT THE PROJECT", "Practical AI for plant health research.", "PlantHealth AI combines image classification with a transparent, safety-first decision support demonstration.")
    x, y = st.columns(2)
    with x:
        st.markdown('<div class="surface"><div class="surface-title">What the system does</div><div class="muted">A MobileNetV2 transfer-learning model ranks supported plant-health classes from a leaf image. An OpenCV color heuristic estimates a possible affected-color area. A software-only rule module displays the corresponding simulated review status.</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="surface"><div class="surface-title">Project workflow</div><div class="muted">Image upload → disease-class prediction → experimental visual estimate → simulated sprinkler status → human review</div></div>', unsafe_allow_html=True)
    with y:
        st.markdown('<div class="surface"><div class="surface-title">Responsible use</div><div class="muted">The severity proxy is not validated against lesion masks. The test set uses controlled PlantVillage images. The simulation does not control a pump or valve and does not suggest pesticide doses. Treat results as a research demonstration only.</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="surface"><div class="surface-title">Technology</div><div class="muted">Python · Streamlit · TensorFlow / Keras · MobileNetV2 · OpenCV · scikit-learn</div></div>', unsafe_allow_html=True)


def main():
    init_state()
    classes = get_classes()
    report = get_report()
    page = render_sidebar(classes, report)

    if page == "Overview":
        render_overview(classes, report)
    elif page == "Disease Detection":
        render_detection(classes)
    elif page == "Performance":
        render_performance(report)
    elif page == "Plant Library":
        render_library(classes)
    else:
        render_about()

    st.markdown('<div class="footer-note">PlantHealth AI · Research prototype · Image predictions and experimental severity estimates require human review.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
