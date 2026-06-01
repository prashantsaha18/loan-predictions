"""
train_model.py
==============
Standalone script to train, tune, and serialize the best loan-approval model.

Usage:
    python train_model.py

Output:
    models/best_model.pkl – serialised champion pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_preprocessor import load_and_split
from src.model_trainer import LoanModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the full ML pipeline: load → split → train → evaluate → save."""

    csv_path = str(ROOT / "data" / "loan_data.csv")
    model_dir = ROOT / "models"

    logger.info("=" * 60)
    logger.info("Loan Approval Prediction – Model Training")
    logger.info("=" * 60)

    # 1. Load and split data
    logger.info("Loading dataset from '%s' …", csv_path)
    X_train, X_test, y_train, y_test = load_and_split(csv_path)

    # 2. Train all models
    trainer = LoanModelTrainer(model_dir=model_dir, n_iter=15, cv=5)
    results = trainer.train(X_train, X_test, y_train, y_test)

    # 3. Print comparison table
    comparison_df = trainer.get_comparison_df()
    logger.info("\n%s", comparison_df.to_string(index=False))

    # 4. Save best model
    save_path = trainer.save_best_model()
    logger.info("✅  Champion model saved to '%s'.", save_path)

    # 5. Feature importance
    fi = trainer.get_feature_importance()
    if fi is not None:
        logger.info("\nTop-10 Features:\n%s", fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
