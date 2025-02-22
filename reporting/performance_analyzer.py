import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from settings.logging_config import logger

class PerformanceAnalyzer:
    def __init__(self, reports_dir='reports'):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.template_path = self.reports_dir / 'performance_report.md'
        self._ensure_template_exists()
    
    def _ensure_template_exists(self):
        """Create template file if it doesn't exist."""
        if not self.template_path.exists():
            template_content = """# Vote Efficiency Model Performance Report
Generated on: {datetime}

## Overall Performance Metrics

### Classification Performance
- Overall Accuracy: {overall_accuracy:.4f}
- Precision Score: {precision_score:.4f}
- Recall Score: {recall_score:.4f}
- F1 Score: {f1_score:.4f}

### Efficiency Prediction Performance
- Mean Absolute Error (MAE): {mae:.4f}
- Root Mean Square Error (RMSE): {rmse:.4f}

## Detailed Analysis

### Vote Decision Analysis
- True Positives: {tp} (Correctly predicted successful votes)
- True Negatives: {tn} (Correctly predicted unsuccessful votes)
- False Positives: {fp} (Incorrectly predicted successful votes)
- False Negatives: {fn} (Incorrectly predicted unsuccessful votes)

### Efficiency Statistics
- Mean Predicted Efficiency: {mean_pred_eff:.2f}%
- Median Predicted Efficiency: {median_pred_eff:.2f}%
- Standard Deviation: {std_pred_eff:.2f}%

### Vote Timing Analysis
- Average Optimal Delay: {avg_optimal_delay:.2f} minutes
- Median Optimal Delay: {median_optimal_delay:.2f} minutes

---
*Report generated automatically by Vote Efficiency Model*
"""
            self.template_path.write_text(template_content)
            logger.info(f"Created performance report template at: {self.template_path}")

    def analyze_performance(self, prediction_df):
        """Analyze and generate detailed performance results."""
        try:
            metrics = self._calculate_metrics(prediction_df)
            report_content = self.template_path.read_text().format(**metrics)
            
            # Save the report with timestamp
            report_path = self.reports_dir / f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
            report_path.write_text(report_content)
            
            logger.info(f"Detailed performance report saved to: {report_path}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing performance results: {e}")
            raise

    def _calculate_metrics(self, prediction_df):
        """Calculate all metrics for the performance report."""
        # Calculate basic metrics
        overall_accuracy = accuracy_score(
            prediction_df['real_success'], 
            prediction_df['vote_decision']
        )
        
        mae = np.mean(np.abs(
            prediction_df['like_efficiency'] - 
            prediction_df['predicted_efficiency']
        ))
        
        rmse = np.sqrt(np.mean(
            (prediction_df['like_efficiency'] - 
             prediction_df['predicted_efficiency'])**2
        ))
        
        # Classification metrics
        precision = precision_score(
            prediction_df['real_success'], 
            prediction_df['vote_decision']
        )
        recall = recall_score(
            prediction_df['real_success'], 
            prediction_df['vote_decision']
        )
        f1 = f1_score(
            prediction_df['real_success'], 
            prediction_df['vote_decision']
        )
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            prediction_df['real_success'], 
            prediction_df['vote_decision']
        ).ravel()
        
        # Additional statistics
        efficiency_stats = prediction_df['predicted_efficiency'].describe()
        timing_stats = prediction_df['optimal_vote_delay_minutes'].describe()
        
        return {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_accuracy': overall_accuracy,
            'precision_score': precision,
            'recall_score': recall,
            'f1_score': f1,
            'mae': mae,
            'rmse': rmse,
            'tp': tp, 
            'tn': tn, 
            'fp': fp, 
            'fn': fn,
            'mean_pred_eff': efficiency_stats['mean'],
            'median_pred_eff': efficiency_stats['50%'],
            'std_pred_eff': efficiency_stats['std'],
            'avg_optimal_delay': timing_stats['mean'],
            'median_optimal_delay': timing_stats['50%']
        }