from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ..settings.logger_config import logger
from ..settings.plot_config import configure_plotting_style

class PerformanceAnalyzer:
    def __init__(self, predictions_df: pd.DataFrame, metrics: Dict):
        self.predictions = predictions_df
        self.metrics = metrics
        self.figures_path = 'reports/figures'
        configure_plotting_style()  # Configure plotting style on initialization

    def generate_performance_report(self, output_path: str) -> None:
        """Generate comprehensive performance report."""
        report_sections = [
            self._generate_metrics_summary(),
            self._generate_predictions_analysis(),
            self._generate_recommendations()
        ]
        
        full_report = "\n\n".join(report_sections)
        
        with open(output_path, 'w') as f:
            f.write(full_report)
        
        logger.info(f"Performance report saved to {output_path}")

    def _generate_metrics_summary(self) -> str:
        """Generate summary of model metrics."""
        summary = "# Model Performance Metrics\n\n"
        
        if 'classifier_metrics' in self.metrics:
            summary += "## Classification Metrics\n"
            for metric, value in self.metrics['classifier_metrics'].items():
                summary += f"- {metric.upper()}: {value:.4f}\n"
        
        if 'regressor_metrics' in self.metrics:
            summary += "\n## Regression Metrics\n"
            for metric, value in self.metrics['regressor_metrics'].items():
                summary += f"- {metric.upper()}: {value:.4f}\n"
        
        return summary

    def plot_efficiency_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot distribution of vote efficiency."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.predictions, x='predicted_efficiency', bins=30)
        plt.title('Distribution of Predicted Vote Efficiency')
        plt.xlabel('Efficiency (%)')
        plt.ylabel('Count')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Efficiency distribution plot saved to {save_path}")
        else:
            plt.show()

    def plot_delay_vs_efficiency(self, save_path: Optional[str] = None) -> None:
        """Plot relationship between vote delay and efficiency."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.predictions, 
                       x='optimal_vote_delay_minutes', 
                       y='predicted_efficiency')
        plt.title('Vote Delay vs Predicted Efficiency')
        plt.xlabel('Vote Delay (minutes)')
        plt.ylabel('Efficiency (%)')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Delay vs efficiency plot saved to {save_path}")
        else:
            plt.show()

    def generate_author_insights(self) -> pd.DataFrame:
        """Generate insights about author performance."""
        insights = pd.DataFrame({
            'Author': self.predictions['Author'].unique()
        })
        
        insights['avg_efficiency'] = self.predictions.groupby('Author')['predicted_efficiency'].mean()
        insights['optimal_delay'] = self.predictions.groupby('Author')['optimal_vote_delay_minutes'].mean()
        insights['success_rate'] = (self.predictions.groupby('Author')['vote_decision'].mean() * 100)
        
        return insights.sort_values('avg_efficiency', ascending=False)

    def _generate_predictions_analysis(self) -> str:
        """Generate analysis of model predictions."""
        analysis = "# Prediction Analysis\n\n"
        
        # Add statistics about predictions
        pos_votes = (self.predictions['vote_decision'] == 1).sum()
        total_pred = len(self.predictions)
        
        analysis += f"## Summary Statistics\n"
        analysis += f"- Total Predictions: {total_pred}\n"
        analysis += f"- Positive Vote Decisions: {pos_votes} ({pos_votes/total_pred*100:.1f}%)\n"
        analysis += f"- Average Predicted Efficiency: {self.predictions['predicted_efficiency'].mean():.2f}%\n"
        
        return analysis

    def _generate_recommendations(self) -> str:
        """Generate actionable recommendations."""
        recommendations = "# Recommendations\n\n"
        
        # Add model-based recommendations
        high_eff_authors = self.predictions[
            self.predictions['predicted_efficiency'] > self.predictions['predicted_efficiency'].mean()
        ]['Author'].unique()
        
        recommendations += "## Top Author Recommendations\n"
        for author in high_eff_authors[:5]:
            author_data = self.predictions[self.predictions['Author'] == author]
            avg_eff = author_data['predicted_efficiency'].mean()
            opt_delay = author_data['optimal_vote_delay_minutes'].mean()
            
            recommendations += f"- {author}:\n"
            recommendations += f"  - Average Efficiency: {avg_eff:.2f}%\n"
            recommendations += f"  - Optimal Voting Delay: {opt_delay:.0f} minutes\n"
        
        return recommendations