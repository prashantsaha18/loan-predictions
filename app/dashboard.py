"""
dashboard.py
============
Streamlit tab renderers for the EDA and Model Performance views.
Each public function corresponds to one dashboard tab.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from app.components import (
    PALETTE,
    metric_card,
    plot_confusion_matrix,
    plot_feature_distribution,
    plot_feature_importance,
    plot_model_comparison,
)


# ---------------------------------------------------------------------------
# Tab 1 – Dataset Overview & EDA
# ---------------------------------------------------------------------------


def render_eda_tab(df: pd.DataFrame) -> None:
    """Render the Exploratory Data Analysis tab.

    Args:
        df: Raw (unprocessed) loan dataset loaded from CSV.
    """
    st.markdown("## 📊 Dataset Overview")

    total       = len(df)
    approved    = int((df["Loan_Status"] == "Y").sum()) if "Loan_Status" in df.columns else 0
    rejected    = total - approved
    missing_pct = df.isnull().mean().mean() * 100
    features    = df.shape[1] - 1  # exclude target

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Total Records", f"{total:,}")
    with col2:
        metric_card("Approved", f"{approved:,}", f"{approved / total * 100:.1f}%")
    with col3:
        metric_card("Rejected", f"{rejected:,}", f"{rejected / total * 100:.1f}%")
    with col4:
        metric_card("Features", str(features))
    with col5:
        metric_card("Missing Values", f"{missing_pct:.1f}%")

    st.markdown("---")

    # ── Feature distributions ──────────────────────────────────────────────
    st.markdown("### Feature Distributions")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Credit History vs. Loan Status**")
        if "Credit_History" in df.columns:
            ch_df = df.dropna(subset=["Credit_History"]).copy()
            ch_df["Credit_History"] = ch_df["Credit_History"].map(
                {1.0: "Good", 0.0: "Poor"}
            )
            st.plotly_chart(
                plot_feature_distribution(ch_df, "Credit_History", color_by="Loan_Status"),
                use_container_width=True,
            )

    with col_b:
        st.markdown("**Property Area vs. Loan Status**")
        if "Property_Area" in df.columns:
            st.plotly_chart(
                plot_feature_distribution(df, "Property_Area", color_by="Loan_Status"),
                use_container_width=True,
            )

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Applicant Income Distribution**")
        if "ApplicantIncome" in df.columns:
            q99 = df["ApplicantIncome"].quantile(0.99)
            income_df = df[df["ApplicantIncome"] < q99]
            st.plotly_chart(
                plot_feature_distribution(income_df, "ApplicantIncome", color_by="Loan_Status"),
                use_container_width=True,
            )

    with col_d:
        st.markdown("**Loan Amount Distribution**")
        if "LoanAmount" in df.columns:
            la_df = df.dropna(subset=["LoanAmount"])
            st.plotly_chart(
                plot_feature_distribution(la_df, "LoanAmount", color_by="Loan_Status"),
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("### Education & Marital Status")
    col_e, col_f = st.columns(2)
    with col_e:
        if "Education" in df.columns:
            st.plotly_chart(
                plot_feature_distribution(df, "Education", color_by="Loan_Status"),
                use_container_width=True,
            )
    with col_f:
        if "Married" in df.columns:
            st.plotly_chart(
                plot_feature_distribution(df, "Married", color_by="Loan_Status"),
                use_container_width=True,
            )

    # ── Correlation heatmap ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔥 Correlation Heatmap (Numerical Features)")
    _render_correlation_heatmap(df)

    # ── Raw data preview ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Raw Data Preview")
    st.dataframe(df.head(50), use_container_width=True, height=300)


def _render_correlation_heatmap(df: pd.DataFrame) -> None:
    """Draw an annotated correlation heatmap for numerical columns.

    Args:
        df: Source DataFrame (may contain non-numeric columns; they are dropped).
    """
    import plotly.graph_objects as go

    num_df = df.select_dtypes(include="number").copy()
    if "Loan_Status" not in num_df.columns and "Loan_Status" in df.columns:
        num_df["Loan_Status"] = (df["Loan_Status"] == "Y").astype(int)

    corr = num_df.corr().round(2)
    cols = corr.columns.tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=cols,
            y=cols,
            colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#6366f1"]],
            zmid=0,
            text=corr.values,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=PALETTE["text"],
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 – Model Performance
# ---------------------------------------------------------------------------


def render_model_tab(
    model_bundle: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Render the Model Performance tab.

    Displays the 3-model comparison chart and table (read from the stored
    ``all_results`` in the pickle), plus the confusion matrix and feature
    importance of the champion model.

    Args:
        model_bundle: Dict loaded from ``best_model.pkl``.
        X_test: Test feature matrix (for confusion matrix).
        y_test: True test labels.
    """
    st.markdown("## 🏆 Model Performance")

    best_name   = model_bundle.get("model_name", "Unknown")
    best_metrics = model_bundle.get("metrics", {})
    all_results  = model_bundle.get("all_results", [best_metrics])

    # ── Champion badge ─────────────────────────────────────────────────────
    prim = PALETTE["primary"]
    sec  = PALETTE["secondary"]
    st.markdown(
        f"""
        <div style="display:inline-block;
                    background:linear-gradient(90deg,{prim},{sec});
                    border-radius:999px;padding:0.4rem 1.4rem;margin-bottom:1rem;">
            <span style="color:white;font-weight:700;">🥇 Champion Model: {best_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Champion metric cards ──────────────────────────────────────────────
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    cols = st.columns(5)
    for col, key in zip(cols, metric_keys):
        with col:
            metric_card(key, f"{best_metrics.get(key, 0):.4f}")

    st.markdown("---")

    # ── 3-Model Comparison (from stored all_results) ───────────────────────
    st.markdown("### 📊 All Models – Side-by-Side Comparison")
    comparison_df = pd.DataFrame(all_results)

    # Styled table
    st.dataframe(
        comparison_df.style
        .highlight_max(
            subset=metric_keys,
            color="#6366f133",
            axis=0,
        )
        .format({k: "{:.4f}" for k in metric_keys}),
        use_container_width=True,
        hide_index=True,
        height=140,
    )

    # Grouped bar chart
    st.plotly_chart(
        plot_model_comparison(comparison_df),
        use_container_width=True,
    )

    st.markdown("---")

    # ── Confusion matrix + feature importance ─────────────────────────────
    pipeline = model_bundle["pipeline"]
    y_pred   = pipeline.predict(X_test)
    cm       = confusion_matrix(y_test, y_pred)

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown(f"### Confusion Matrix — {best_name}")
        st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

    with col_fi:
        st.markdown("### Feature Importance")
        estimator   = pipeline.named_steps["clf"]
        preprocessor = pipeline.named_steps["preprocessor"]

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None

        if hasattr(estimator, "feature_importances_") and feature_names is not None:
            fi_df = (
                pd.DataFrame({
                    "Feature":    feature_names,
                    "Importance": estimator.feature_importances_,
                })
                .sort_values("Importance", ascending=False)
            )
            st.plotly_chart(plot_feature_importance(fi_df), use_container_width=True)

        elif hasattr(estimator, "coef_") and feature_names is not None:
            fi_df = (
                pd.DataFrame({
                    "Feature":    feature_names,
                    "Importance": np.abs(estimator.coef_[0]),
                })
                .sort_values("Importance", ascending=False)
            )
            st.plotly_chart(plot_feature_importance(fi_df), use_container_width=True)

        else:
            st.info("Feature importance not available for this model type.")

    st.markdown("---")

    # ── Best hyper-parameters ──────────────────────────────────────────────
    st.markdown("### ⚙️ Champion Hyper-Parameters")
    best_params = model_bundle.get("best_params", {})
    if best_params:
        st.dataframe(
            pd.DataFrame(
                [{"Parameter": k, "Value": str(v)} for k, v in best_params.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No hyper-parameter info available.")


# ---------------------------------------------------------------------------
# Tab 4 – Batch Prediction
# ---------------------------------------------------------------------------


def render_batch_tab(model_bundle: Dict[str, Any]) -> None:
    """Render the Batch Prediction tab.

    Users upload a CSV that matches the training schema (minus Loan_Status).
    The app returns a downloadable CSV with Prediction and Confidence columns
    appended.

    Args:
        model_bundle: Dict loaded from ``best_model.pkl``.
    """
    st.markdown("## 📂 Batch Prediction")
    st.markdown(
        "<p style='color:#94a3b8;'>Upload a CSV file with applicant records. "
        "The system will predict approval status for every row and return a "
        "downloadable results file.</p>",
        unsafe_allow_html=True,
    )

    # ── Template download ──────────────────────────────────────────────────
    template_cols = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area",
    ]
    template_df = pd.DataFrame(columns=template_cols)
    template_csv = template_df.to_csv(index=False)

    st.download_button(
        label="⬇️  Download CSV Template",
        data=template_csv,
        file_name="loan_batch_template.csv",
        mime="text/csv",
    )

    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload applicant CSV",
        type=["csv"],
        help="File must contain the same columns as the training dataset (Loan_Status is optional and will be ignored).",
    )

    if uploaded is None:
        st.info("Upload a CSV file above to begin batch scoring.")
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return

    # Drop columns we don't need
    batch_df = batch_df.drop(columns=["Loan_ID", "Loan_Status"], errors="ignore")

    st.markdown(f"**Preview** — {len(batch_df):,} rows × {len(batch_df.columns)} columns")
    st.dataframe(batch_df.head(10), use_container_width=True, height=240)

    if st.button("▶️  Run Batch Prediction", use_container_width=True):
        with st.spinner(f"Scoring {len(batch_df):,} applicants…"):
            try:
                pipeline = model_bundle["pipeline"]
                preds    = pipeline.predict(batch_df)
                probas   = pipeline.predict_proba(batch_df)

                results_df = batch_df.copy()
                results_df["Prediction"]  = ["Approved" if p == 1 else "Rejected" for p in preds]
                results_df["Confidence"]  = [
                    round(float(probas[i][p]) * 100, 1) for i, p in enumerate(preds)
                ]

                approved_count = int((preds == 1).sum())
                rejected_count = int((preds == 0).sum())

                # Summary cards
                c1, c2, c3 = st.columns(3)
                with c1:
                    metric_card("Total Scored", f"{len(results_df):,}")
                with c2:
                    metric_card("Approved", f"{approved_count:,}",
                                f"{approved_count / len(results_df) * 100:.1f}%")
                with c3:
                    metric_card("Rejected", f"{rejected_count:,}",
                                f"{rejected_count / len(results_df) * 100:.1f}%")

                st.markdown("### Results Preview")
                st.dataframe(results_df, use_container_width=True, height=340)

                # Download results
                result_csv = results_df.to_csv(index=False)
                st.download_button(
                    label="⬇️  Download Full Results CSV",
                    data=result_csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")
