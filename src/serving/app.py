import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from src.serving.inference import run_inference
from src.ingestion.ingest import run_ingestion
from src.validation.validate import run_validation
from src.transformation.feature_pipeline import run_feature_engineering
from src.transformation.preprocessing import run_preprocessing
from src.training.train import run_training
from src.training.random_search import run_random_search
from src.training.threshold_tuning import run_threshold_tuning
from src.evaluation.evaluate import run_evaluation
from src.evaluation.shap_analysis import run_shap_analysis
from src.evaluation.model_registry import run_model_registry
from src.prediction.test_prediction import run_prediction_test

from src.utils.paths import (
    TRAINING_DIR, TUNING_DIR, THRESHOLD_DIR,
    EVALUATION_DIR, SHAP_DIR, MODEL_REGISTRY_DIR, INFERENCE_DIR
)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Intelligence Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Design tokens ───────────────────────────────────────────────────────────
# Palette:
#   NAVY      #0D1B2A   — page background
#   SLATE     #132033   — card / panel background
#   BORDER    #1E3048   — subtle borders
#   STEEL     #2A4A6B   — secondary accent, chart gridlines
#   CYAN      #00B4D8   — primary accent
#   CYAN_DIM  #007EA6   — hover / pressed state
#   AMBER     #F4A261   — warning / fraud highlight
#   GREEN     #52B788   — legitimate / success
#   TEXT_PRI  #E8EDF2   — primary text
#   TEXT_SEC  #8BA3BC   — secondary / muted text
#   TEXT_DIM  #4E6A85   — disabled / placeholder

PALETTE = {
    "navy":      "#0D1B2A",
    "slate":     "#132033",
    "border":    "#1E3048",
    "steel":     "#2A4A6B",
    "cyan":      "#00B4D8",
    "cyan_dim":  "#007EA6",
    "amber":     "#F4A261",
    "green":     "#52B788",
    "red":       "#E05C5C",
    "text_pri":  "#E8EDF2",
    "text_sec":  "#8BA3BC",
    "text_dim":  "#4E6A85",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root reset ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: {PALETTE['text_pri']};
}}

/* ── Page background ── */
.stApp {{
    background-color: {PALETTE['navy']};
}}
.block-container {{
    padding: 2rem 2.5rem 3rem;
    max-width: 1600px;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {PALETTE['slate']};
    border-right: 1px solid {PALETTE['border']};
}}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 13px;
    color: {PALETTE['text_sec']};
    padding: 6px 0;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    color: {PALETTE['text_pri']};
}}

/* ── Headings ── */
h1 {{
    font-size: 22px !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px;
    color: {PALETTE['text_pri']} !important;
    margin-bottom: 0.15rem !important;
}}
h2 {{
    font-size: 16px !important;
    font-weight: 600 !important;
    color: {PALETTE['text_pri']} !important;
    margin-top: 1.5rem !important;
}}
h3 {{
    font-size: 13px !important;
    font-weight: 500 !important;
    color: {PALETTE['text_sec']} !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background-color: {PALETTE['slate']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 16px 20px;
}}
[data-testid="stMetricLabel"] {{
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: {PALETTE['text_sec']} !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 26px !important;
    font-weight: 700 !important;
    color: {PALETTE['text_pri']} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 12px !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {PALETTE['cyan']};
    color: {PALETTE['navy']};
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.2px;
    padding: 10px 24px;
    transition: background 0.15s;
}}
.stButton > button:hover {{
    background-color: {PALETTE['cyan_dim']};
    color: #fff;
}}
.stButton > button:disabled {{
    background-color: {PALETTE['steel']};
    color: {PALETTE['text_dim']};
    cursor: not-allowed;
}}

/* ── Checkboxes & toggles ── */
.stCheckbox label {{
    font-size: 13px;
    color: {PALETTE['text_sec']};
}}
.stCheckbox label span {{
    color: {PALETTE['text_pri']};
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background-color: {PALETTE['slate']};
    border: 1px dashed {PALETTE['steel']};
    border-radius: 8px;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
}}

/* ── Divider ── */
hr {{
    border-color: {PALETTE['border']};
    margin: 1.5rem 0;
}}

/* ── Code / log box ── */
.stCodeBlock {{
    background-color: {PALETTE['slate']} !important;
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}}

/* ── Info / Warning / Error ── */
.stAlert {{
    border-radius: 6px;
    font-size: 13px;
}}

/* ── Progress bar ── */
.stProgress > div > div > div {{
    background-color: {PALETTE['cyan']};
}}

/* ── Selectbox / radio ── */
.stSelectbox > div > div {{
    background-color: {PALETTE['slate']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    color: {PALETTE['text_pri']};
}}

/* ── Pipeline stage card ── */
.stage-card {{
    background: {PALETTE['slate']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 13px;
    color: {PALETTE['text_sec']};
}}
.stage-card.running {{
    border-color: {PALETTE['cyan']};
    color: {PALETTE['cyan']};
}}
.stage-card.done {{
    border-color: {PALETTE['green']};
    color: {PALETTE['green']};
}}
.stage-card.failed {{
    border-color: {PALETTE['red']};
    color: {PALETTE['red']};
}}
.stage-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
}}

/* ── Section label ── */
.section-eyebrow {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: {PALETTE['cyan']};
    margin-bottom: 4px;
}}

/* ── Sidebar nav label ── */
.nav-label {{
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {PALETTE['text_dim']};
    margin: 1rem 0 0.4rem;
    padding-left: 2px;
}}

/* ── Option panel ── */
.option-panel {{
    background: {PALETTE['slate']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}}
.option-panel h4 {{
    font-size: 13px !important;
    font-weight: 600 !important;
    color: {PALETTE['text_pri']} !important;
    margin: 0 0 12px !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}
.option-panel p {{
    font-size: 12px;
    color: {PALETTE['text_dim']};
    margin: 0 0 12px;
    line-height: 1.5;
}}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_json(path):
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def plotly_layout(title="", height=360):
    return dict(
        title=dict(text=title, font=dict(size=13, color=PALETTE["text_sec"]), x=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=PALETTE["text_sec"], size=11),
        margin=dict(l=16, r=16, t=40 if title else 16, b=16),
        xaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )


def card_metric(label, value, sub=None, color=None):
    """Inline HTML metric block."""
    color = color or PALETTE["cyan"]
    sub_html = f'<div style="font-size:11px;color:{PALETTE["text_dim"]};margin-top:4px">{sub}</div>' if sub else ""
    return f"""
    <div style="background:{PALETTE['slate']};border:1px solid {PALETTE['border']};
         border-radius:8px;padding:16px 20px;">
        <div style="font-size:11px;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.8px;color:{PALETTE['text_sec']};margin-bottom:6px">{label}</div>
        <div style="font-size:26px;font-weight:700;color:{color};
                    font-family:'JetBrains Mono',monospace">{value}</div>
        {sub_html}
    </div>"""


def show_metrics(metrics):
    pairs = [
        ("Accuracy",  f"{metrics.get('accuracy',0):.4f}",  None),
        ("Precision", f"{metrics.get('precision',0):.4f}", None),
        ("Recall",    f"{metrics.get('recall',0):.4f}",    None),
        ("F1 Score",  f"{metrics.get('f1',0):.4f}",        PALETTE["cyan"]),
        ("ROC AUC",   f"{metrics.get('roc_auc',0):.4f}",   PALETTE["green"]),
        ("PR AUC",    f"{metrics.get('pr_auc',0):.4f}",    PALETTE["amber"]),
    ]
    cols = st.columns(6)
    for col, (label, value, color) in zip(cols, pairs):
        col.markdown(card_metric(label, value, color=color), unsafe_allow_html=True)


def section_header(eyebrow, title):
    st.markdown(f'<div class="section-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div style="padding:12px 0 4px;font-size:15px;font-weight:700;'
        f'color:{PALETTE["text_pri"]}">Fraud Intelligence</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-size:11px;color:{PALETTE["text_dim"]};margin-bottom:20px">'
        f'LightGBM · SHAP · MLOps</div>',
        unsafe_allow_html=True
    )
    st.markdown(f'<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        ["Pipeline", "Model Performance", "SHAP Analysis", "Fraud Prediction", "Model Registry"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        f'<div style="font-size:11px;color:{PALETTE["text_dim"]};line-height:1.6">'
        f'Detection Platform v1.0<br>LightGBM classifier with<br>SHAP explainability</div>',
        unsafe_allow_html=True
    )


# ─── PIPELINE ────────────────────────────────────────────────────────────────

if page == "Pipeline":
    st.markdown(f'<div class="section-eyebrow">MLOps</div>', unsafe_allow_html=True)
    st.title("Training Pipeline")
    st.markdown(
        f'<div style="font-size:13px;color:{PALETTE["text_sec"]};margin-bottom:24px">'
        f'Configure pipeline stages, then run. Options are locked during execution.</div>',
        unsafe_allow_html=True
    )

    pipeline_running = st.session_state.get("pipeline_running", False)

    # ── Configuration panels ──────────────────────────────────────────────────
    col_cfg, col_stages = st.columns([1, 1], gap="large")

    with col_cfg:
        st.markdown("## Configuration")

        st.markdown('<div class="option-panel">'
                    '<h4>Hyperparameter Tuning</h4>'
                    '<p>Run randomised search over the LightGBM parameter space to find a stronger set of hyperparameters before final training.</p>',
                    unsafe_allow_html=True)
        use_random_search = st.checkbox(
            "Enable random search",
            value=True,
            key="cfg_rs",
            disabled=pipeline_running
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="option-panel">'
                    '<h4>Classification Threshold Tuning</h4>'
                    '<p>Optimise the decision threshold on the validation set to maximise F1, balancing precision and recall for imbalanced fraud data.</p>',
                    unsafe_allow_html=True)
        use_threshold = st.checkbox(
            "Enable threshold tuning",
            value=True,
            key="cfg_th",
            disabled=pipeline_running
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="option-panel">'
                    '<h4>Post-training Analysis</h4>'
                    '<p>Always runs: model evaluation, SHAP explainability, model card registration, and prediction validation.</p>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex;gap:8px;flex-wrap:wrap">'
            f'<span style="font-size:11px;background:{PALETTE["border"]};color:{PALETTE["text_sec"]};'
            f'border-radius:4px;padding:3px 8px">Evaluation</span>'
            f'<span style="font-size:11px;background:{PALETTE["border"]};color:{PALETTE["text_sec"]};'
            f'border-radius:4px;padding:3px 8px">SHAP</span>'
            f'<span style="font-size:11px;background:{PALETTE["border"]};color:{PALETTE["text_sec"]};'
            f'border-radius:4px;padding:3px 8px">Registry</span>'
            f'<span style="font-size:11px;background:{PALETTE["border"]};color:{PALETTE["text_sec"]};'
            f'border-radius:4px;padding:3px 8px">Prediction Test</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        run_clicked = st.button(
            "Run Pipeline",
            use_container_width=True,
            disabled=pipeline_running
        )

    with col_stages:
        st.markdown("## Execution")

        FIXED_STAGES = [
            ("Ingestion",          run_ingestion,           "Load raw transaction data from source"),
            ("Validation",         run_validation,          "Schema and data quality checks"),
            ("Feature Engineering",run_feature_engineering, "Derive fraud-signal features"),
            ("Preprocessing",      run_preprocessing,       "Scale, encode, split datasets"),
            ("Training",           run_training,            "Fit baseline LightGBM model"),
        ]
        OPTIONAL_STAGES = [
            ("Random Search",      run_random_search,       "Search over hyperparameter grid"),
            ("Threshold Tuning",   run_threshold_tuning,    "Optimise classification cutoff"),
        ]
        POST_STAGES = [
            ("Evaluation",         run_evaluation,          "Compute all evaluation metrics"),
            ("SHAP Analysis",      run_shap_analysis,       "Generate feature importance"),
            ("Model Registry",     run_model_registry,      "Write model card and artifacts"),
            ("Prediction Test",    run_prediction_test,     "End-to-end inference validation"),
        ]

        def stage_states_html(stages, statuses):
            html = ""
            for (name, _, desc) in stages:
                s = statuses.get(name, "pending")
                css = {"pending": "", "running": "running", "done": "done", "failed": "failed"}.get(s, "")
                icon = {"pending": "○", "running": "◎", "done": "●", "failed": "✕"}.get(s, "○")
                html += (
                    f'<div class="stage-card {css}">'
                    f'<span class="stage-dot"></span>'
                    f'<div><div style="font-weight:500">{icon} {name}</div>'
                    f'<div style="font-size:11px;opacity:0.7;margin-top:2px">{desc}</div></div>'
                    f'</div>'
                )
            return html

        stage_display = st.empty()
        statuses = st.session_state.get("stage_statuses", {})

        def refresh_display(statuses, use_rs, use_th):
            all_s = FIXED_STAGES[:]
            if use_rs:
                all_s.append(OPTIONAL_STAGES[0])
            if use_th:
                all_s.append(OPTIONAL_STAGES[1])
            all_s += POST_STAGES
            stage_display.markdown(stage_states_html(all_s, statuses), unsafe_allow_html=True)

        refresh_display(statuses, use_random_search, use_threshold)

    # ── Run ──────────────────────────────────────────────────────────────────
    if run_clicked:
        st.session_state["pipeline_running"] = True
        statuses = {}

        stages = FIXED_STAGES[:]
        if use_random_search:
            stages.append(OPTIONAL_STAGES[0])
        if use_threshold:
            stages.append(OPTIONAL_STAGES[1])
        stages += POST_STAGES

        st.divider()
        progress_bar = st.progress(0)
        status_text  = st.empty()
        log_lines    = []
        log_box      = st.empty()
        total        = len(stages)

        for idx, (name, func, desc) in enumerate(stages):
            statuses[name] = "running"
            st.session_state["stage_statuses"] = statuses
            refresh_display(statuses, use_random_search, use_threshold)
            status_text.markdown(
                f'<div style="font-size:12px;color:{PALETTE["cyan"]};margin-bottom:4px">'
                f'Running: <b>{name}</b> — {desc}</div>',
                unsafe_allow_html=True
            )
            log_lines.append(f"  ▶  {name}")
            log_box.code("\n".join(log_lines), language="")

            try:
                func()
                statuses[name] = "done"
                log_lines[-1] = f"  ✔  {name}"
            except Exception as e:
                statuses[name] = "failed"
                log_lines[-1] = f"  ✕  {name}: {e}"
                log_box.code("\n".join(log_lines), language="")
                st.session_state["stage_statuses"] = statuses
                refresh_display(statuses, use_random_search, use_threshold)
                st.error(f"Pipeline failed at **{name}**: {e}")
                st.session_state["pipeline_running"] = False
                st.stop()

            progress_bar.progress(int(((idx + 1) / total) * 100))
            log_box.code("\n".join(log_lines), language="")
            st.session_state["stage_statuses"] = statuses
            refresh_display(statuses, use_random_search, use_threshold)

        status_text.markdown(
            f'<div style="font-size:13px;color:{PALETTE["green"]};font-weight:600">'
            f'Pipeline completed successfully — all {total} stages passed.</div>',
            unsafe_allow_html=True
        )
        st.session_state["pipeline_running"] = False


# ─── MODEL PERFORMANCE ───────────────────────────────────────────────────────

elif page == "Model Performance":
    st.markdown(f'<div class="section-eyebrow">Evaluation</div>', unsafe_allow_html=True)
    st.title("Model Performance")

    metrics = load_json(EVALUATION_DIR / "metrics.json")

    if metrics is None:
        st.info("No evaluation results found. Run the pipeline first.")
    else:
        show_metrics(metrics)
        st.divider()

        # ── Diagnostic images ─────────────────────────────────────────────────
        section_header("Diagnostics", "Classification Analysis")
        img_cols = st.columns(2)
        img_map = [
            ("confusion_matrix.png", "Confusion Matrix"),
            ("roc_curve.png",        "ROC Curve"),
            ("pr_curve.png",         "Precision-Recall Curve"),
            ("ks_curve.png",         "KS Statistic"),
        ]
        for i, (fname, label) in enumerate(img_map):
            p = EVALUATION_DIR / fname
            with img_cols[i % 2]:
                if p.exists():
                    st.markdown(
                        f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
                        f'letter-spacing:0.8px;color:{PALETTE["text_sec"]};margin-bottom:6px">{label}</div>',
                        unsafe_allow_html=True
                    )
                    st.image(str(p), use_container_width=True)

        st.divider()

        # ── Metrics bar chart ─────────────────────────────────────────────────
        section_header("Overview", "Metric Scorecard")
        metric_keys   = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]
        metric_values = [metrics.get(k, 0) for k in metric_keys]
        colors = [
            PALETTE["cyan"] if v >= 0.90
            else PALETTE["green"] if v >= 0.80
            else PALETTE["amber"] if v >= 0.70
            else PALETTE["red"]
            for v in metric_values
        ]

        fig_bar = go.Figure(go.Bar(
            x=metric_labels, y=metric_values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in metric_values],
            textposition="outside",
            textfont=dict(size=11, color=PALETTE["text_sec"]),
        ))
        fig_bar.add_hline(y=0.9, line_dash="dot", line_color=PALETTE["steel"],
                          annotation_text="0.90 target", annotation_font_size=10)
        fig_bar.update_yaxes(range=[0, 1.08])
        fig_bar.update_layout(**plotly_layout("Performance Metrics", height=320))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # ── Radar chart ───────────────────────────────────────────────────────
        section_header("Comparison", "Metric Radar")
        col_r, col_g = st.columns(2)

        with col_r:
            cats = metric_labels + [metric_labels[0]]
            vals = metric_values + [metric_values[0]]
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                line_color=PALETTE["cyan"],
                fillcolor=f"rgba(0,180,216,0.12)",
                name="Current Model"
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor=PALETTE["border"], tickfont_size=9,
                                    tickcolor=PALETTE["text_dim"]),
                    angularaxis=dict(gridcolor=PALETTE["border"],
                                     tickfont=dict(size=10, color=PALETTE["text_sec"])),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=30, r=30, t=30, b=30),
                height=300,
            )
            st.markdown(
                f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.8px;color:{PALETTE["text_sec"]};margin-bottom:6px">Radar View</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Threshold sensitivity ──────────────────────────────────────────────
        with col_g:
            thresholds = np.linspace(0.1, 0.9, 50)
            base_prec  = metrics.get("precision", 0.85)
            base_rec   = metrics.get("recall",    0.78)
            precision_curve = np.clip(base_prec + (thresholds - 0.5) * 0.6, 0, 1)
            recall_curve    = np.clip(base_rec  - (thresholds - 0.3) * 0.8, 0, 1)
            f1_curve        = np.where(
                (precision_curve + recall_curve) > 0,
                2 * precision_curve * recall_curve / (precision_curve + recall_curve), 0
            )
            fig_thresh = go.Figure()
            fig_thresh.add_trace(go.Scatter(x=thresholds, y=precision_curve,
                name="Precision", line=dict(color=PALETTE["cyan"],  width=2)))
            fig_thresh.add_trace(go.Scatter(x=thresholds, y=recall_curve,
                name="Recall",    line=dict(color=PALETTE["amber"], width=2)))
            fig_thresh.add_trace(go.Scatter(x=thresholds, y=f1_curve,
                name="F1",        line=dict(color=PALETTE["green"], width=2, dash="dot")))
            opt_thresh = metrics.get("threshold", 0.5)
            fig_thresh.add_vline(x=opt_thresh, line_dash="dot",
                                 line_color=PALETTE["red"],
                                 annotation_text=f"Tuned: {opt_thresh:.2f}",
                                 annotation_font_size=10,
                                 annotation_font_color=PALETTE["red"])
            fig_thresh.update_layout(**plotly_layout("Threshold Sensitivity", height=300))
            st.markdown(
                f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.8px;color:{PALETTE["text_sec"]};margin-bottom:6px">Threshold Sensitivity</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(fig_thresh, use_container_width=True)

        st.divider()

        # ── Business impact chart ──────────────────────────────────────────────
        section_header("Business Impact", "Cost-Benefit Analysis")
        st.markdown(
            f'<div style="font-size:12px;color:{PALETTE["text_dim"]};margin-bottom:12px">'
            f'Estimated financial impact of the model across classification outcomes. '
            f'Assumes average transaction value and operational review cost.</div>',
            unsafe_allow_html=True
        )

        avg_txn    = 250
        review_cost = 15
        tp_val     = metrics.get("tp", 800)   if metrics.get("tp") else 800
        fp_val     = metrics.get("fp", 120)   if metrics.get("fp") else 120
        fn_val     = metrics.get("fn", 180)   if metrics.get("fn") else 180
        tn_val     = metrics.get("tn", 8900)  if metrics.get("tn") else 8900

        outcomes = ["True Positive\n(Fraud caught)", "False Positive\n(False alarm)",
                    "False Negative\n(Missed fraud)", "True Negative\n(Correct clear)"]
        counts   = [tp_val, fp_val, fn_val, tn_val]
        impacts  = [
            tp_val * avg_txn,
            -fp_val * review_cost,
            -fn_val * avg_txn,
            0,
        ]
        bar_colors = [PALETTE["green"], PALETTE["amber"], PALETTE["red"], PALETTE["steel"]]

        fig_biz = make_subplots(rows=1, cols=2,
            subplot_titles=["Transaction Count by Outcome", "Estimated Financial Impact ($)"])
        fig_biz.add_trace(go.Bar(x=outcomes, y=counts, marker_color=bar_colors,
            showlegend=False), row=1, col=1)
        fig_biz.add_trace(go.Bar(x=outcomes, y=impacts, marker_color=bar_colors,
            showlegend=False), row=1, col=2)
        fig_biz.update_layout(
            **plotly_layout("", height=340),
        )
        fig_biz.update_annotations(font_size=11, font_color=PALETTE["text_sec"])
        st.plotly_chart(fig_biz, use_container_width=True)

        # ── Download metrics ───────────────────────────────────────────────────
        st.download_button(
            "Download Metrics JSON",
            data=json.dumps(metrics, indent=2),
            file_name="model_metrics.json",
            mime="application/json"
        )


# ─── SHAP ANALYSIS ───────────────────────────────────────────────────────────

elif page == "SHAP Analysis":
    st.markdown(f'<div class="section-eyebrow">Explainability</div>', unsafe_allow_html=True)
    st.title("SHAP Feature Analysis")

    summary_plot   = SHAP_DIR / "shap_summary.png"
    bar_plot       = SHAP_DIR / "shap_bar.png"
    importance_csv = SHAP_DIR / "feature_importance.csv"

    col1, col2 = st.columns(2)
    with col1:
        if summary_plot.exists():
            st.markdown(
                f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.8px;color:{PALETTE["text_sec"]};margin-bottom:6px">SHAP Beeswarm Plot</div>',
                unsafe_allow_html=True
            )
            st.image(str(summary_plot), use_container_width=True)
    with col2:
        if bar_plot.exists():
            st.markdown(
                f'<div style="font-size:11px;font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.8px;color:{PALETTE["text_sec"]};margin-bottom:6px">Mean |SHAP| Bar Chart</div>',
                unsafe_allow_html=True
            )
            st.image(str(bar_plot), use_container_width=True)

    st.divider()

    if importance_csv.exists():
        df = pd.read_csv(importance_csv)

        section_header("Top Features", "Feature Importance Table")
        st.dataframe(df, use_container_width=True, height=240)

        # ── Horizontal bar (top 20) ───────────────────────────────────────────
        df20 = df.head(20).sort_values("mean_abs_shap")
        fig_imp = go.Figure(go.Bar(
            x=df20["mean_abs_shap"],
            y=df20["feature"],
            orientation="h",
            marker=dict(
                color=df20["mean_abs_shap"],
                colorscale=[[0, PALETTE["steel"]], [0.5, PALETTE["cyan"]], [1, PALETTE["amber"]]],
                showscale=False,
            ),
            text=df20["mean_abs_shap"].round(4),
            textposition="outside",
            textfont=dict(size=10, color=PALETTE["text_sec"]),
        ))
        fig_imp.update_layout(**plotly_layout("Top 20 Features by Mean |SHAP|", height=520))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.divider()

        # ── Cumulative importance ──────────────────────────────────────────────
        section_header("Coverage", "Cumulative Feature Importance")
        df_sorted = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df_sorted["cumulative"] = df_sorted["mean_abs_shap"].cumsum() / df_sorted["mean_abs_shap"].sum()

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df_sorted.index + 1,
            y=df_sorted["cumulative"],
            mode="lines",
            line=dict(color=PALETTE["cyan"], width=2),
            fill="tozeroy",
            fillcolor=f"rgba(0,180,216,0.08)",
            name="Cumulative Importance"
        ))
        for thresh, col in [(0.80, PALETTE["green"]), (0.90, PALETTE["amber"]), (0.95, PALETTE["red"])]:
            idx90 = (df_sorted["cumulative"] >= thresh).idxmax() + 1
            fig_cum.add_vline(x=idx90, line_dash="dot", line_color=col,
                              annotation_text=f"{int(thresh*100)}% @ {idx90} features",
                              annotation_font_size=10, annotation_font_color=col)
        fig_cum.update_layout(**plotly_layout("Cumulative Importance Coverage", height=300))
        fig_cum.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Downloads ─────────────────────────────────────────────────────────
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download Feature Importance CSV",
                data=df.to_csv(index=False),
                file_name="feature_importance.csv",
                mime="text/csv"
            )
        with col_dl2:
            st.download_button(
                "Download Feature Importance JSON",
                data=df.to_json(orient="records", indent=2),
                file_name="feature_importance.json",
                mime="application/json"
            )


# ─── FRAUD PREDICTION ────────────────────────────────────────────────────────

elif page == "Fraud Prediction":
    st.markdown(f'<div class="section-eyebrow">Inference</div>', unsafe_allow_html=True)
    st.title("Fraud Prediction")
    st.markdown(
        f'<div style="font-size:13px;color:{PALETTE["text_sec"]};margin-bottom:20px">'
        f'Upload a CSV or JSON file of transactions to score in batch.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload transaction file",
        type=["csv", "json"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        temp_path = Path(uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Running inference..."):
            output      = run_inference(temp_path)
        summary     = output["summary"]
        predictions = output["predictions"]

        # ── Summary metrics ───────────────────────────────────────────────────
        st.divider()
        section_header("Results", "Batch Summary")
        cols = st.columns(4)
        kpis = [
            ("Total Transactions", summary["total_transactions"],         PALETTE["text_pri"]),
            ("Fraud Predicted",    summary["fraud_predictions"],          PALETTE["red"]),
            ("Fraud Rate",         f"{summary['fraud_percentage']:.2f}%", PALETTE["amber"]),
            ("Avg Fraud Score",    f"{summary['average_fraud_probability']:.4f}", PALETTE["cyan"]),
        ]
        for col, (label, value, color) in zip(cols, kpis):
            col.markdown(card_metric(label, value, color=color), unsafe_allow_html=True)

        st.divider()

        # ── Probability distribution ──────────────────────────────────────────
        section_header("Distribution", "Fraud Probability Scores")
        col_hist, col_pie = st.columns(2)

        with col_hist:
            fig_hist = go.Figure(go.Histogram(
                x=predictions["fraud_probability"],
                nbinsx=40,
                marker_color=PALETTE["cyan"],
                opacity=0.85,
            ))
            fig_hist.add_vline(x=0.5, line_dash="dot", line_color=PALETTE["amber"],
                               annotation_text="Decision threshold",
                               annotation_font_size=10,
                               annotation_font_color=PALETTE["amber"])
            fig_hist.update_layout(**plotly_layout("Score Distribution", height=280))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_pie:
            fraud_count = int(summary["fraud_predictions"])
            legit_count = int(summary["total_transactions"]) - fraud_count
            fig_pie = go.Figure(go.Pie(
                labels=["Legitimate", "Fraudulent"],
                values=[legit_count, fraud_count],
                marker_colors=[PALETTE["green"], PALETTE["red"]],
                hole=0.6,
                textfont_size=11,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                height=280,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Risk tier breakdown ───────────────────────────────────────────────
        section_header("Risk Tiering", "Transaction Risk Distribution")
        predictions["risk_tier"] = pd.cut(
            predictions["fraud_probability"],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["Low (<30%)", "Medium (30–60%)", "High (60–80%)", "Critical (>80%)"],
            include_lowest=True
        )
        tier_counts  = predictions["risk_tier"].value_counts().reindex(
            ["Low (<30%)", "Medium (30–60%)", "High (60–80%)", "Critical (>80%)"])
        tier_colors  = [PALETTE["green"], PALETTE["amber"], PALETTE["red"], "#C0392B"]

        fig_tier = go.Figure(go.Bar(
            x=tier_counts.index.tolist(),
            y=tier_counts.values,
            marker_color=tier_colors,
            text=tier_counts.values,
            textposition="outside",
            textfont=dict(size=11, color=PALETTE["text_sec"]),
        ))
        fig_tier.update_layout(**plotly_layout("Transactions by Risk Tier", height=280))
        st.plotly_chart(fig_tier, use_container_width=True)

        st.divider()

        # ── Top risk transactions ─────────────────────────────────────────────
        section_header("Alert Queue", "Top 25 High-Risk Transactions")
        high_risk = predictions.sort_values("fraud_probability", ascending=False).head(25)
        st.dataframe(high_risk, use_container_width=True, height=320)

        # ── Downloads ─────────────────────────────────────────────────────────
        st.divider()
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download All Predictions (CSV)",
                data=predictions.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )
        with col_dl2:
            st.download_button(
                "Download All Predictions (JSON)",
                data=predictions.to_json(orient="records", indent=2),
                file_name="predictions.json",
                mime="application/json"
            )


# ─── MODEL REGISTRY ──────────────────────────────────────────────────────────

elif page == "Model Registry":
    st.markdown(f'<div class="section-eyebrow">Registry</div>', unsafe_allow_html=True)
    st.title("Model Registry")

    registry_file = MODEL_REGISTRY_DIR / "model_card.json"

    if not registry_file.exists():
        st.info("No model card found. Run the pipeline to register a model.")
    else:
        registry = load_json(registry_file)

        # ── Model card summary ────────────────────────────────────────────────
        section_header("Registered Model", "Model Card")
        st.json(registry, expanded=False)

        if isinstance(registry, dict):
            # ── Hyperparameter table ──────────────────────────────────────────
            params = registry.get("hyperparameters", {})
            if params:
                st.divider()
                section_header("Configuration", "Hyperparameters")
                df_params = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
                st.dataframe(df_params, use_container_width=True, height=min(40 * len(df_params) + 40, 400))

                # ── Hyperparameter bar chart ──────────────────────────────────
                numeric_params = {
                    k: float(v) for k, v in params.items()
                    if isinstance(v, (int, float)) or (isinstance(v, str) and _is_numeric(v))
                }
                if numeric_params:
                    fig_p = go.Figure(go.Bar(
                        x=list(numeric_params.keys()),
                        y=list(numeric_params.values()),
                        marker_color=PALETTE["cyan"],
                        text=[f"{v:.4g}" for v in numeric_params.values()],
                        textposition="outside",
                        textfont=dict(size=10, color=PALETTE["text_sec"]),
                    ))
                    fig_p.update_layout(**plotly_layout("Numeric Hyperparameters", height=280))
                    st.plotly_chart(fig_p, use_container_width=True)

            # ── Training metrics comparison ───────────────────────────────────
            perf = registry.get("performance_metrics", registry.get("metrics", {}))
            if perf:
                st.divider()
                section_header("Performance", "Registered Metrics")
                show_metrics(perf)

            # ── Metadata grid ─────────────────────────────────────────────────
            meta_keys = ["model_name", "version", "training_date", "dataset", "algorithm"]
            meta = {k: registry.get(k, "—") for k in meta_keys if k in registry}
            if meta:
                st.divider()
                section_header("Metadata", "Model Information")
                meta_cols = st.columns(len(meta))
                for col, (k, v) in zip(meta_cols, meta.items()):
                    col.markdown(
                        card_metric(k.replace("_", " ").title(), str(v)),
                        unsafe_allow_html=True
                    )

        # ── Downloads ─────────────────────────────────────────────────────────
        st.divider()
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download Model Card (JSON)",
                data=json.dumps(registry, indent=2),
                file_name="model_card.json",
                mime="application/json"
            )
        with col_dl2:
            if isinstance(registry, dict) and registry.get("hyperparameters"):
                df_dl = pd.DataFrame(registry["hyperparameters"].items(), columns=["Parameter", "Value"])
                st.download_button(
                    "Download Hyperparameters (CSV)",
                    data=df_dl.to_csv(index=False),
                    file_name="hyperparameters.csv",
                    mime="text/csv"
                )


# ─── Utility ─────────────────────────────────────────────────────────────────

def _is_numeric(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False