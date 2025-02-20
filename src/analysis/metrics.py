from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
from ..settings.logger_config import logger

class ModelMetrics:
    def __init__(self):
        self.classifier_metrics = {}
        self.regressor_metrics = {}

    def calculate_classifier_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics."""
        self.classifier_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        return self.classifier_metrics

    def calculate_regressor_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        self.regressor_metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        return self.regressor_metrics

    def log_metrics(self) -> None:
        """Log all calculated metrics."""
        if self.classifier_metrics:
            logger.info("Classification Metrics:")
            for metric, value in self.classifier_metrics.items():
                logger.info(f"{metric.upper()}: {value:.4f}")

        if self.regressor_metrics:
            logger.info("Regression Metrics:")
            for metric, value in self.regressor_metrics.items():
                logger.info(f"{metric.upper()}: {value:.4f}")