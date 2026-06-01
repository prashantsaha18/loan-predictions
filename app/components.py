"""
components.py
=============
Reusable Streamlit UI widgets for the Loan Approval dashboard.
All components are pure functions; state lives in the calling page.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Shared colour palette
# ---------------------------------------------------------------------------

PALETTE: Dict[str, str] = {
    "approved":  "#10b981",
    "rejected":  "#ef4444",
    "primary":   "#6366f1",
    "secondary": "#f59e0b",
    "bg_dark":   "#0f172a",
    "bg_card":   "#1e293b",
    "text":      "#f1f5f9",
    "muted":     "#94a3b8",
}


# ---------------------------------------------------------------------------
# Metric / status widgets
# ---------------------------------------------------------------------------


def metric_card(
    label: str,
    value: str | int | float,
    delta: Optional[str] = None,
) -> None:
    """Render a dark-themed metric card.

    Args:
        label: Upper-case caption above the value.
        value: Primary figure to display (large type).
        delta: Optional small annotation below the value.
    """
    muted = PALETTE["muted"]
    delta_html = (
        f'<span style="font-size:0.85rem;color:{muted};">{delta}</span>'
        if delta else ""
    )
    bg   = PALETTE["bg_card"]
    prim = PALETTE["primary"]
    txt  = PALETTE["text"]
    st.markdown(
        f"""
        <div style="background:{bg};border-radius:12px;padding:1.2rem 1.5rem;
                    border-left:4px solid {prim};margin-bottom:0.5rem;">
            <div style="color:{muted};font-size:0.8rem;letter-spacing:0.1em;
                        text-transform:uppercase;">{label}</div>
            <div style="color:{txt};font-size:2rem;font-weight:700;
                        line-height:1.1;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def approval_banner(approved: bool, confidence: float) -> None:
    """Render a full-width approval or rejection result banner.

    Args:
        approved: True → approved (green); False → rejected (red).
        confidence: Model probability for the predicted class (0–1).
    """
    label  = "✅  LOAN APPROVED" if approved else "❌  LOAN REJECTED"
    color  = PALETTE["approved"] if approved else PALETTE["rejected"]
    sub    = (
        "Congratulations! Your application meets our criteria."
        if approved
        else "Unfortunately, this application does not meet our current criteria."
    )
    conf_pct = f"{confidence * 100:.1f}%"
    txt = PALETTE["text"]
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{color}22,{color}11);
                    border:2px solid {color};border-radius:16px;
                    padding:2rem;text-align:center;margin:1rem 0;">
            <div style="color:{color};font-size:2.2rem;font-weight:900;
                        letter-spacing:0.04em;">{label}</div>
            <div style="color:{txt};font-size:1rem;margin-top:0.5rem;">{sub}</div>
            <div style="margin-top:1.2rem;">
                <span style="background:{color};color:white;border-radius:999px;
                             padding:0.4rem 1.2rem;font-size:1.1rem;font-weight:700;">
                    Confidence: {conf_pct}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Loan application form
# ---------------------------------------------------------------------------


def loan_application_form() -> Tuple[Optional[Dict[str, Any]], bool]:
    """Render the loan application input form.

    Returns:
        Tuple of (input_dict, submitted).
        *input_dict* is ``None`` until the form is submitted.
    """
    with st.form("loan_form", clear_on_submit=False):
        st.markdown("#### 👤 Applicant Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox(
                "Gender", ["Male", "Female"],
                help="Applicant's gender identity.",
            )
        with c2:
            married = st.selectbox(
                "Marital Status", ["Yes", "No"],
                help="Is the applicant currently married?",
            )
        with c3:
            dependents = st.selectbox(
                "Dependents", ["0", "1", "2", "3+"],
                help="Number of financially supported dependents.",
            )

        c4, c5 = st.columns(2)
        with c4:
            education = st.selectbox(
                "Education Level", ["Graduate", "Not Graduate"],
                help="Highest level of education attained.",
            )
        with c5:
            self_employed = st.selectbox(
                "Self-Employed", ["No", "Yes"],
                help="Is the applicant self-employed?",
            )

        st.markdown("---")
        st.markdown("#### 💰 Financial Details")
        c6, c7 = st.columns(2)
        with c6:
            applicant_income = st.number_input(
                "Applicant Monthly Income (₹)",
                min_value=0, value=5000, step=500,
                help="Gross monthly income of the primary applicant.",
            )
        with c7:
            coapplicant_income = st.number_input(
                "Co-Applicant Monthly Income (₹)",
                min_value=0, value=0, step=500,
                help="Monthly income of a co-applicant (0 if none).",
            )

        c8, c9 = st.columns(2)
        with c8:
            loan_amount = st.number_input(
                "Requested Loan Amount (₹ thousands)",
                min_value=1, value=150, step=10,
                help="Desired loan amount in thousands of Rupees.",
            )
        with c9:
            loan_term = st.selectbox(
                "Loan Term (months)",
                [36, 60, 84, 120, 180, 240, 300, 360, 480],
                index=7,
                help="Repayment period in months (360 = 30 years).",
            )

        st.markdown("---")
        st.markdown("#### 🏠 Property & Credit")
        c10, c11 = st.columns(2)
        with c10:
            credit_history = st.selectbox(
                "Credit History",
                options=[1, 0],
                format_func=lambda x: (
                    "Good (repaid debts)" if x == 1 else "Poor (defaults on record)"
                ),
                help="Whether the applicant has repaid all previous debts.",
            )
        with c11:
            property_area = st.selectbox(
                "Property Area", ["Urban", "Semiurban", "Rural"],
                help="Location type of the property to be purchased.",
            )

        submitted = st.form_submit_button(
            "🔍  Submit Application", use_container_width=True
        )

    if submitted:
        return {
            "Gender":            gender,
            "Married":           married,
            "Dependents":        dependents,
            "Education":         education,
            "Self_Employed":     self_employed,
            "ApplicantIncome":   applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount":        loan_amount,
            "Loan_Amount_Term":  loan_term,
            "Credit_History":    float(credit_history),
            "Property_Area":     property_area,
        }, True
    return None, False


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def plot_feature_distribution(
    df: pd.DataFrame,
    column: str,
    color_by: Optional[str] = None,
) -> go.Figure:
    """Histogram or grouped bar chart for a single feature.

    Args:
        df: Source DataFrame.
        column: Column to visualise.
        color_by: Optional column used for colour grouping.

    Returns:
        Plotly Figure.
    """
    if df[column].dtype == "object" or df[column].nunique() <= 6:
        fig = px.histogram(
            df, x=column, color=color_by, barmode="group",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_dark",
        )
    else:
        fig = px.histogram(
            df, x=column, color=color_by, nbins=30,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_dark",
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=PALETTE["text"],
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_confusion_matrix(
    cm: Any,
    labels: Optional[List[str]] = None,
) -> go.Figure:
    """Interactive heatmap for a 2×2 confusion matrix.

    Args:
        cm: 2-D array-like confusion matrix.
        labels: Class label strings (default: Rejected / Approved).

    Returns:
        Plotly Figure.
    """
    if labels is None:
        labels = ["Rejected (0)", "Approved (1)"]
    cm_arr = np.array(cm)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr, x=labels, y=labels,
            colorscale=[[0, "#1e293b"], [1, PALETTE["primary"]]],
            text=cm_arr, texttemplate="%{text}", showscale=False,
        )
    )
    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=PALETTE["text"],
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of feature importances.

    Args:
        importance_df: DataFrame with ``Feature`` and ``Importance`` columns.
        top_n: Maximum number of features to display.

    Returns:
        Plotly Figure.
    """
    df = importance_df.head(top_n).sort_values("Importance")
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=["#6366f1", "#f59e0b"],
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=PALETTE["text"], coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing metrics across models.

    Args:
        comparison_df: DataFrame with columns Model, Accuracy, Precision,
            Recall, F1-Score, ROC-AUC.

    Returns:
        Plotly Figure.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors  = [
        PALETTE["primary"], PALETTE["secondary"], PALETTE["approved"],
        "#ec4899", "#06b6d4",
    ]
    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=comparison_df["Model"],
                y=comparison_df[metric],
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color=PALETTE["text"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(range=[0, 1.05]),
    )
    return fig
