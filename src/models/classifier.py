from ..settings.logger_config import logger
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

class VoteClassifier:
    def __init__(self, random_state=42):
        self.model = XGBClassifier(random_state=random_state)
        self.feature_columns = ['vote_delay', 'author_avg_efficiency', 
                              'author_reputation', 'author_avg_payout']
    
    def prepare_data(self, df):
        """Prepare features and target for classification."""
        X = df[self.feature_columns]
        y = df['success']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train(self, X_train, y_train):
        """Train the classifier model."""
        logger.info("Training vote classifier model...")
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the classifier model."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Classifier Accuracy Score: {accuracy:.4f}")
        logger.info("Classification Report:\n" + report)
        
        return accuracy, report
    
    def predict(self, features):
        """Predict whether to vote on a post."""
        return self.model.predict(features)
    
    def save_model(self, filepath='models/classifier_model.json'):
        """Save the trained model."""
        self.model.save_model(filepath)
        logger.info(f"Classifier model saved to {filepath}")
    
    def load_model(self, filepath='models/classifier_model.json'):
        """Load a trained model."""
        self.model.load_model(filepath)
        logger.info(f"Classifier model loaded from {filepath}")