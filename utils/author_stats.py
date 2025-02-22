import pandas as pd

def generate_author_statistics(df):
    """Generate comprehensive author statistics."""
    author_stats = pd.DataFrame({
        'Author': df['Author'].unique(),
        'Total_Posts': df.groupby('Author').size(),
        'Avg_Efficiency': df.groupby('Author')['like_efficiency'].mean(),
        'Success_Rate': df.groupby('Author')['success'].mean() * 100,
        'Avg_Payout': df.groupby('Author')['author_avg_payout'].mean(),
        'Avg_Vote_Delay': df.groupby('Author')['vote_delay'].mean(),
        'Reputation': df.groupby('Author')['author_reputation'].first()
    }).reset_index(drop=True)
    
    return author_stats