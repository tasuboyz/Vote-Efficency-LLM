import pandas as pd
import os
from settings.logging_config import logger

class ExcelReporter:
    def __init__(self, base_path, curator):
        self.base_path = base_path
        self.curator = curator
        self.filepath = os.path.join(base_path, f'model_performance_{curator}.xlsx')
    
    def save_to_excel(self, data_dict):
        """
        Save multiple dataframes to Excel sheets
        data_dict: Dictionary with sheet_name: dataframe pairs
        """
        with pd.ExcelWriter(self.filepath) as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"Excel file saved successfully at: {self.filepath}")
    
    def prepare_rankings_data(self, author_stats):
        """Prepare different rankings from author statistics."""
        return {
            'Top Authors by Efficiency': author_stats.nlargest(10, 'Avg_Efficiency'),
            'Bottom Authors by Efficiency': author_stats.nsmallest(10, 'Avg_Efficiency'),
            'Top Authors by Success Rate': author_stats.nlargest(10, 'Success_Rate'),
            'Bottom Authors by Success Rate': author_stats.nsmallest(10, 'Success_Rate'),
            'Top Authors by Payout': author_stats.nlargest(10, 'Avg_Payout'),
            'Bottom Authors by Payout': author_stats.nsmallest(10, 'Avg_Payout'),
            'Complete Author Stats': author_stats.sort_values('Avg_Efficiency', ascending=False)
        }
    
    def save_prediction_reports(self, prediction_df, author_stats):
        """Save prediction results and author statistics."""
        data_dict = {
            'Predictions': prediction_df,
            **self.prepare_rankings_data(author_stats)
        }
        self.save_to_excel(data_dict)
    
    def save_production_report(self, prediction_df):
        """Save production predictions."""
        production_data = {
            'Production Predictions': prediction_df[
                ['Post', 'Author', 'vote_decision', 
                 'optimal_vote_delay_minutes', 'predicted_efficiency']
            ]
        }
        self.save_to_excel(production_data)