from ..settings.logger_config import logger
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class EfficiencyRegressor:
    def __init__(self, random_state=42):
        self.model = XGBRegressor(random_state=random_state)
        self.feature_columns = ['vote_delay', 'author_avg_efficiency', 
                              'author_reputation', 'author_avg_payout']
    
    def prepare_data(self, df):
        """Prepare features and target for regression."""
        X = df[self.feature_columns]
        y = df['like_efficiency']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train(self, X_train, y_train):
        """Train the regressor model."""
        logger.info("Training efficiency regressor model...")
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the regressor model."""
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Regressor MSE: {mse:.4f}")
        logger.info(f"Regressor R2 Score: {r2:.4f}")
        
        return mse, r2
    
    def predict(self, features):
        """Predict the efficiency score."""
        return self.model.predict(features)
    
    def save_model(self, filepath='models/regressor_model.json'):
        """Save the trained model."""
        self.model.save_model(filepath)
        logger.info(f"Regressor model saved to {filepath}")
    
    def load_model(self, filepath='models/regressor_model.json'):
        """Load a trained model."""
        self.model.load_model(filepath)
        logger.info(f"Regressor model loaded from {filepath}")