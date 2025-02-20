import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from ..settings.logger_config import logger

class DataProcessor:
    def __init__(self, raw_data: Dict):
        self.raw_data = raw_data
        self.df = pd.DataFrame(raw_data)
        self.optimal_delay_history = None

    def prepare_training_data(self) -> Tuple:
        """Prepare data for model training."""
        # Calculate optimal delay history
        self.optimal_delay_history = self._calculate_optimal_delays()
        
        # Prepare features and targets
        X = self.df[['vote_delay', 'author_avg_efficiency', 
                     'author_reputation', 'author_avg_payout']]
        y_clf = self.df['success']
        y_reg = self.df['like_efficiency']
        
        # Split data
        X_train, X_test, y_clf_train, y_clf_test = train_test_split(
            X, y_clf, test_size=0.2, random_state=42
        )
        
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        return (X_train, X_test, y_clf_train, y_clf_test, 
                X_reg_train, X_reg_test, y_reg_train, y_reg_test)

    def _calculate_optimal_delays(self) -> Dict:
        """Calculate optimal voting delays for each author."""
        return (self.df[self.df['success'] == 1]
                .groupby('Author')['vote_delay']
                .mean()
                .to_dict())

    def generate_author_statistics(self) -> pd.DataFrame:
        """Generate author performance statistics."""
        author_stats = pd.DataFrame({
            'Author': self.df['Author'].unique(),
            'Total_Posts': self.df.groupby('Author').size(),
            'Avg_Efficiency': self.df.groupby('Author')['like_efficiency'].mean(),
            'Success_Rate': self.df.groupby('Author')['success'].mean() * 100,
            'Avg_Payout': self.df.groupby('Author')['author_avg_payout'].mean()
        }).reset_index(drop=True)
        
        return author_stats

    def save_results_to_excel(self, 
                            prediction_df: pd.DataFrame, 
                            curator: str,
                            filename: Optional[str] = None) -> None:
        """Save all results to Excel file."""
        if filename is None:
            filename = f'predictions_and_rankings_{curator}.xlsx'
            
        author_stats = self.generate_author_statistics()
        
        with pd.ExcelWriter(filename) as writer:
            # Save predictions
            prediction_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Save rankings
            self._save_rankings(writer, author_stats)
            
            # Save complete stats
            author_stats.sort_values('Avg_Efficiency', ascending=False).to_excel(
                writer, 
                sheet_name='Complete Author Stats', 
                index=False
            )
        
        logger.info(f"Results saved to {filename}")

    def _save_rankings(self, writer, author_stats: pd.DataFrame) -> None:
        """Save various author rankings to Excel sheets."""
        rankings = {
            'Efficiency': 'Avg_Efficiency',
            'Success Rate': 'Success_Rate',
            'Payout': 'Avg_Payout'
        }
        
        for name, column in rankings.items():
            author_stats.nlargest(10, column).to_excel(
                writer, 
                sheet_name=f'Top Authors by {name}', 
                index=False
            )
            author_stats.nsmallest(10, column).to_excel(
                writer, 
                sheet_name=f'Bottom Authors by {name}', 
                index=False
            )