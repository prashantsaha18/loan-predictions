"""
app.py
======
Streamlit entry point for the Loan Approval Prediction System.

Run locally:  streamlit run app.py
Deploy:       push repo to GitHub → Streamlit Cloud → main file: app.py

On first launch, if ``models/best_model.pkl`` is not found, the app will
automatically train all three models so the dashboard is immediately usable
without any manual pre-training step.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.components import PALETTE, approval_banner, loan_application_form, metric_card
from app.dashboard import render_batch_tab, render_eda_tab, render_model_tab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LoanIQ – Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global styles
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family:'Inter',sans-serif;
                                  background-color:#0f172a; color:#f1f5f9; }
    h1, h2, h3, h4 { font-family:'Syne',sans-serif !important; }
    .stTabs [data-baseweb="tab-list"] { background:#1e293b; border-radius:12px;
                                         padding:4px; gap:4px; }
    .stTabs [data-baseweb="tab"]      { border-radius:8px; font-family:'Syne',sans-serif;
                                         font-weight:600; color:#94a3b8;
                                         padding:0.5rem 1.5rem; }
    .stTabs [aria-selected="true"]    { background:#6366f1 !important; color:white !important; }
    .stButton > button { background:linear-gradient(135deg,#6366f1,#8b5cf6); color:white;
                         border:none; border-radius:10px; font-family:'Syne',sans-serif;
                         font-weight:700; font-size:1rem; padding:0.75rem 2rem;
                         transition:opacity 0.2s; }
    .stButton > button:hover { opacity:0.88; }
    .stSelectbox > div, .stNumberInput > div { border-radius:8px; }
    div[data-testid="stDownloadButton"] > button {
        background:linear-gradient(135deg,#0f172a,#1e293b);
        border:1px solid #6366f1; color:#6366f1; }
    #MainMenu { visibility:hidden; } footer { visibility:hidden; } header { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data & model loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading dataset…")
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the loan CSV with Streamlit caching.

    Args:
        csv_path: Path to loan_data.csv.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the file is absent.
    """
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        st.error(f"Dataset not found at: {csv_path}")
        raise exc


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: str) -> Optional[Dict[str, Any]]:
    """Deserialise the champion model pipeline from pickle.

    Args:
        model_path: Path to best_model.pkl.

    Returns:
        Model bundle dict or None if absent / corrupt.
    """
    try:
        with open(model_path, "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Auto-training (runs once on first deploy if pkl is missing)
# ---------------------------------------------------------------------------


def _auto_train(data_path: str, model_path: str) -> Optional[Dict[str, Any]]:
    """Train all models and save the champion if no pkl exists yet.

    Displays a Streamlit progress UI while training runs so Streamlit Cloud
    users see live feedback rather than a blank screen.

    Args:
        data_path: Path to the CSV dataset.
        model_path: Destination path for best_model.pkl.

    Returns:
        Loaded model bundle dict, or None on failure.
    """
    from src.data_preprocessor import load_and_split
    from src.model_trainer import LoanModelTrainer

    status = st.status("🤖 First run detected — training models…", expanded=True)
    try:
        with status:
            st.write("📂 Loading & splitting dataset…")
            X_train, X_test, y_train, y_test = load_and_split(data_path)

            st.write("⚙️  Training Logistic Regression, Random Forest, Gradient Boosting…")
            trainer = LoanModelTrainer(
                model_dir=Path(model_path).parent,
                n_iter=10,   # fewer iters keeps startup time under ~30 s on cloud
                cv=5,
            )
            trainer.train(X_train, X_test, y_train, y_test)

            st.write("💾 Saving champion model…")
            trainer.save_best_model()

            df = trainer.get_comparison_df()
            st.write("✅ Training complete!")
            st.dataframe(df, hide_index=True, use_container_width=True)

        status.update(label="✅ Models trained and ready!", state="complete", expanded=False)

        # Bust the cache so load_model picks up the new file
        load_model.clear()
        return load_model(model_path)

    except Exception as exc:
        status.update(label=f"❌ Training failed: {exc}", state="error")
        logger.error("Auto-training failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_loan(
    model_bundle: Dict[str, Any],
    input_data: Dict[str, Any],
) -> tuple[int, float]:
    """Run inference on a single applicant record.

    Args:
        model_bundle: Loaded model bundle.
        input_data: Dict of raw feature values.

    Returns:
        (prediction: 0|1, confidence: float in [0,1]).

    Raises:
        RuntimeError: If inference fails.
    """
    try:
        pipeline   = model_bundle["pipeline"]
        df         = pd.DataFrame([input_data])
        prediction = int(pipeline.predict(df)[0])
        proba      = float(pipeline.predict_proba(df)[0][prediction])
        return prediction, proba
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit application entry point."""

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:1.5rem 0 0.5rem;">
            <div style="font-size:3.5rem;font-weight:900;font-family:'Syne',sans-serif;
                        background:linear-gradient(135deg,#6366f1,#f59e0b);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                🏦 LoanIQ
            </div>
            <div style="color:#94a3b8;font-size:1.1rem;margin-top:0.2rem;">
                AI-Powered Loan Approval Prediction System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    data_path  = str(ROOT / "data"   / "loan_data.csv")
    model_path = str(ROOT / "models" / "best_model.pkl")

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        st.stop()

    # ── Load or auto-train model ─────────────────────────────────────────────
    model_bundle = load_model(model_path)
    if model_bundle is None:
        model_bundle = _auto_train(data_path, model_path)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_eda, tab_model, tab_portal, tab_batch = st.tabs([
        "📊  Dataset & EDA",
        "🏆  Model Performance",
        "📝  Loan Application",
        "📂  Batch Prediction",
    ])

    # ── Tab 1: EDA ───────────────────────────────────────────────────────────
    with tab_eda:
        render_eda_tab(df)

    # ── Tab 2: Model Performance ─────────────────────────────────────────────
    with tab_model:
        if model_bundle is None:
            st.warning("⚠️  Model unavailable. Check logs above.", icon="⚠️")
        else:
            try:
                from src.data_preprocessor import load_and_split
                _, X_test, _, y_test = load_and_split(data_path)
                render_model_tab(model_bundle, X_test, y_test)
            except Exception as exc:
                st.error(f"Could not render model metrics: {exc}")

    # ── Tab 3: Single Application Portal ─────────────────────────────────────
    with tab_portal:
        st.markdown("## 📝 Loan Application Portal")
        st.markdown(
            "<p style='color:#94a3b8;'>Fill in the form below for an instant "
            "AI-powered approval decision with confidence score.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if model_bundle is None:
            st.warning("⚠️  Model not available.", icon="⚠️")
        else:
            input_data, submitted = loan_application_form()
            if submitted and input_data is not None:
                with st.spinner("Evaluating application…"):
                    try:
                        pred, confidence = predict_loan(model_bundle, input_data)
                        st.markdown("---")
                        st.markdown("### 🔔 Decision")
                        approval_banner(approved=(pred == 1), confidence=confidence)
                        st.markdown("---")
                        st.markdown("### 📋 Application Summary")
                        st.dataframe(
                            pd.DataFrame(
                                [{"Field": k, "Value": str(v)}
                                 for k, v in input_data.items()]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    except RuntimeError as exc:
                        st.error(f"Prediction error: {exc}")

    # ── Tab 4: Batch Prediction ───────────────────────────────────────────────
    with tab_batch:
        if model_bundle is None:
            st.warning("⚠️  Model not available.", icon="⚠️")
        else:
            render_batch_tab(model_bundle)


if __name__ == "__main__":
    main()
