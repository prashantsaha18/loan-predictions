"""
model_trainer.py
================
Trains, tunes, and evaluates three classifiers, then serialises the champion.

Algorithms
----------
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting (HistGradientBoostingClassifier – no external deps)

The model with the highest ROC-AUC on the held-out test set is selected as
champion and bundled with the preprocessor pipeline into ``best_model.pkl``.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.data_preprocessor import build_preprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """Stores evaluation metrics and artefacts for a single trained model.

    Attributes:
        name: Human-readable algorithm name.
        pipeline: Fitted sklearn Pipeline (preprocessor + estimator).
        accuracy: Test-set accuracy score.
        precision: Weighted-average precision.
        recall: Weighted-average recall.
        f1: Weighted-average F1 score.
        roc_auc: Area under the ROC curve.
        best_params: Best hyper-parameters found by RandomizedSearchCV.
    """

    name: str
    pipeline: Pipeline
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    best_params: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return metrics as a plain dictionary (for DataFrame construction)."""
        return {
            "Model": self.name,
            "Accuracy": round(self.accuracy, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1-Score": round(self.f1, 4),
            "ROC-AUC": round(self.roc_auc, 4),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class LoanModelTrainer:
    """Orchestrates training, tuning, and evaluation of loan-approval classifiers.

    Args:
        model_dir: Directory where ``best_model.pkl`` will be written.
        n_iter: Number of parameter settings sampled by RandomizedSearchCV.
        cv: Number of cross-validation folds.
        random_state: Seed for reproducibility.

    Example:
        >>> trainer = LoanModelTrainer(model_dir=Path("models"))
        >>> results = trainer.train(X_train, X_test, y_train, y_test)
        >>> trainer.save_best_model()
    """

    def __init__(
        self,
        model_dir: Path = Path("models"),
        n_iter: int = 15,
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.model_dir = model_dir
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state

        self.results: List[ModelResult] = []
        self.best_result: Optional[ModelResult] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_candidates(self) -> List[Tuple[str, Any, Dict[str, Any]]]:
        """Define three classifier candidates with their hyper-parameter search spaces.

        Returns:
            List of (name, estimator, param_dist) triples ready for
            ``RandomizedSearchCV``.
        """
        lr_candidate = (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            ),
            {
                "clf__C": [0.01, 0.1, 1, 10, 100],
                "clf__solver": ["lbfgs", "liblinear"],
            },
        )

        rf_candidate = (
            "Random Forest",
            RandomForestClassifier(
                random_state=self.random_state,
                class_weight="balanced",
            ),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [None, 5, 10, 20],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        )

        gb_candidate = (
            "Gradient Boosting",
            HistGradientBoostingClassifier(random_state=self.random_state),
            {
                "clf__max_iter": [100, 200, 300],
                "clf__max_depth": [3, 5, 7, None],
                "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "clf__min_samples_leaf": [10, 20, 30],
                "clf__l2_regularization": [0.0, 0.1, 1.0],
            },
        )

        return [lr_candidate, rf_candidate, gb_candidate]

    def _evaluate_pipeline(
        self,
        name: str,
        fitted_pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        best_params: Dict[str, Any],
    ) -> ModelResult:
        """Compute test-set evaluation metrics for a fitted pipeline.

        Args:
            name: Model label.
            fitted_pipeline: Already-fitted sklearn Pipeline.
            X_test: Test feature matrix.
            y_test: True test labels.
            best_params: Best params from randomised search.

        Returns:
            Populated :class:`ModelResult` instance.
        """
        y_pred = fitted_pipeline.predict(X_test)
        y_proba = fitted_pipeline.predict_proba(X_test)[:, 1]

        result = ModelResult(
            name=name,
            pipeline=fitted_pipeline,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
            recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
            f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
            roc_auc=roc_auc_score(y_test, y_proba),
            best_params=best_params,
        )
        logger.info(
            "[%s] Acc=%.4f | F1=%.4f | ROC-AUC=%.4f",
            name, result.accuracy, result.f1, result.roc_auc,
        )
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> List[ModelResult]:
        """Train, tune, and evaluate all candidate models.

        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training labels.
            y_test: Test labels.

        Returns:
            List of :class:`ModelResult` sorted descending by ROC-AUC.
        """
        self.results = []

        for name, estimator, param_dist in self._build_candidates():
            logger.info("Training: %s …", name)

            # Fresh preprocessor per model to avoid data leakage
            preprocessor = build_preprocessor()
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", estimator)])

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=self.n_iter,
                scoring="roc_auc",
                cv=self.cv,
                n_jobs=-1,
                random_state=self.random_state,
                refit=True,
            )
            try:
                search.fit(X_train, y_train)
            except Exception as exc:  # noqa: BLE001
                logger.error("Training failed for %s: %s", name, exc)
                continue

            result = self._evaluate_pipeline(
                name=name,
                fitted_pipeline=search.best_estimator_,
                X_test=X_test,
                y_test=y_test,
                best_params=search.best_params_,
            )
            self.results.append(result)

        self.results.sort(key=lambda r: r.roc_auc, reverse=True)
        self.best_result = self.results[0] if self.results else None

        if self.best_result:
            logger.info(
                "Champion model: %s (ROC-AUC %.4f)",
                self.best_result.name, self.best_result.roc_auc,
            )
        return self.results

    def save_best_model(self, filename: str = "best_model.pkl") -> Path:
        """Serialize the champion pipeline to disk.

        Args:
            filename: Output filename (relative to ``self.model_dir``).

        Returns:
            Absolute path of the saved pickle file.

        Raises:
            RuntimeError: If :meth:`train` has not been called yet.
        """
        if self.best_result is None:
            raise RuntimeError("No trained model found. Call .train() first.")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.model_dir / filename

        with open(save_path, "wb") as fh:
            pickle.dump(
                {
                    "pipeline": self.best_result.pipeline,
                    "model_name": self.best_result.name,
                    "metrics": self.best_result.as_dict(),
                    "best_params": self.best_result.best_params,
                    "all_results": [r.as_dict() for r in self.results],
                },
                fh,
            )
        logger.info("Model saved → '%s'.", save_path)
        return save_path

    def get_comparison_df(self) -> pd.DataFrame:
        """Return a tidy DataFrame of all model metrics.

        Returns:
            DataFrame with one row per model and columns for each metric.
        """
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.as_dict() for r in self.results])

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Extract feature importance from the champion model.

        Returns:
            DataFrame with ``Feature`` and ``Importance`` columns sorted
            descending, or *None* if not available for the model type.
        """
        if self.best_result is None:
            return None

        pipeline = self.best_result.pipeline
        estimator = pipeline.named_steps["clf"]
        preprocessor = pipeline.named_steps["preprocessor"]

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:  # noqa: BLE001
            feature_names = None

        if hasattr(estimator, "feature_importances_") and feature_names is not None:
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_") and feature_names is not None:
            importances = np.abs(estimator.coef_[0])
        else:
            return None

        df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        return df.sort_values("Importance", ascending=False).reset_index(drop=True)
