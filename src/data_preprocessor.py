"""
data_preprocessor.py
====================
Scikit-Learn preprocessing pipeline for the Loan Approval dataset.

Responsibilities
----------------
* Separate numerical vs. categorical feature treatment.
* Impute missing values (median for numerics, most-frequent for categoricals).
* Scale numerical features with StandardScaler.
* Encode categorical features with OneHotEncoder (drop='first' to avoid multicollinearity).
* Expose a single ``build_preprocessor`` factory function consumed by the trainer.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

NUMERICAL_FEATURES: List[str] = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

TARGET_COLUMN: str = "Loan_Status"
DROP_COLUMNS: List[str] = ["Loan_ID"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """Build and return a fitted-ready ColumnTransformer.

    Returns:
        ColumnTransformer: Unfitted preprocessing pipeline combining
        numerical and categorical sub-pipelines.

    Example:
        >>> preprocessor = build_preprocessor()
        >>> X_transformed = preprocessor.fit_transform(X_train)
    """
    numerical_pipeline: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    logger.info("Preprocessor built with %d numerical and %d categorical features.",
                len(NUMERICAL_FEATURES), len(CATEGORICAL_FEATURES))
    return preprocessor


def load_and_split(
    csv_path: str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the CSV, drop irrelevant columns, and split into train/test sets.

    Args:
        csv_path: Absolute or relative path to ``loan_data.csv``.
        test_size: Fraction of data reserved for testing (default 0.20).
        random_state: Reproducibility seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        FileNotFoundError: If the CSV file does not exist at *csv_path*.
        ValueError: If the target column is absent from the dataset.
    """
    from sklearn.model_selection import train_test_split  # local import to keep module light

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        logger.error("Dataset not found at '%s'.", csv_path)
        raise FileNotFoundError(f"Dataset not found: {csv_path}") from exc

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Encode target: Y → 1, N → 0
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Y").astype(int)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Data split – train: %d rows | test: %d rows | positive rate: %.1f%%",
        len(X_train), len(X_test), y.mean() * 100,
    )
    return X_train, X_test, y_train, y_test
