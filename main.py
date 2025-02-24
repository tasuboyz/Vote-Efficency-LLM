from datetime import datetime, timezone
import random
import time
import requests
import json
import os
import pandas as pd
import numpy as np
from beem.account import Account
from beem import Steem, Hive
from beem.nodelist import NodeList
from beem.vote import Vote
from beem.comment import Comment
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, XGBRegressor
from utils.author_database import AuthorDatabase

# Import settings
from settings.config import (
    BLOCKCHAIN_CHOICE, CURATOR, STEEM_NODES, HIVE_NODES,
    MODE_CHOICES, OPERATION_MODE, TEST_SIZE, MAX_RESULTS,
    DIRECTORIES, MODEL_DIR, REPORT_DIR
)
from settings.logging_config import logger
from utils.author_database import AuthorDatabase

def convert_vests_to_power(amount, blockchain_instance):
    """Convert vesting shares to HP/SP in base alla blockchain."""
    try:
        if isinstance(blockchain_instance, Hive):
            return blockchain_instance.vests_to_hp(float(amount))
        elif isinstance(blockchain_instance, Steem):
            return blockchain_instance.vests_to_sp(float(amount))
        else:
            logger.error("Blockchain non supportata")
            return 0
    except Exception as e:
        logger.error(f"Errore nella conversione da vesting shares a power: {e}")
        return 0

def update_efficiency_average(author, current_efficiency, author_efficiency_dict):
    """Aggiorna l'efficienza media di un autore."""
    if author not in author_efficiency_dict:
        author_efficiency_dict[author] = {
            'total': current_efficiency,
            'count': 1,
            'average': current_efficiency
        }
    else:
        author_data = author_efficiency_dict[author]
        author_data['total'] += current_efficiency
        author_data['count'] += 1
        author_data['average'] = author_data['total'] / author_data['count']
    
    return author_efficiency_dict[author]['average']

def update_payout_average(author, current_payout, author_payout_dict):
    """Updates the average payout for an author."""
    if author not in author_payout_dict:
        author_payout_dict[author] = {
            'total': current_payout,
            'count': 1,
            'average': current_payout
        }
    else:
        payout_data = author_payout_dict[author]
        payout_data['total'] += current_payout
        payout_data['count'] += 1
        payout_data['average'] = payout_data['total'] / payout_data['count']
    
    return author_payout_dict[author]['average']

def test_node(node_url, blockchain_type="HIVE"):
    """Test if a node is responsive and functioning correctly."""
    try:
        # Prepare the test request
        headers = {'Content-Type': 'application/json'}
        payload = {
            "jsonrpc": "2.0",
            "method": "condenser_api.get_dynamic_global_properties",
            "params": [],
            "id": 1
        }
        
        # Make request with timeout
        response = requests.post(node_url, json=payload, headers=headers, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            if 'result' in result:
                logger.info(f"Node {node_url} is working properly")
                return True
            
        logger.warning(f"Node {node_url} returned invalid response")
        return False
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Node {node_url} test failed: {str(e)}")
        return False

def get_working_node(blockchain_type="HIVE"):
    """Get a working node from the configured list with fallback mechanism."""
    # Get nodes based on blockchain type
    nodes = HIVE_NODES if blockchain_type == "HIVE" else STEEM_NODES
    working_nodes = []
    
    # Test all nodes and collect working ones
    for node in nodes:
        if test_node(node, blockchain_type):
            working_nodes.append(node)
    
    if not working_nodes:
        raise Exception(f"No working {blockchain_type} nodes found!")
    
    # Return a random working node from the list
    selected_node = random.choice(working_nodes)
    logger.info(f"Selected {blockchain_type} node: {selected_node}")
    return selected_node

def switch_to_backup_node(current_node, blockchain_type="HIVE"):
    """Switch to a different working node, avoiding the current one."""
    nodes = HIVE_NODES if blockchain_type == "HIVE" else STEEM_NODES
    backup_nodes = [node for node in nodes if node != current_node]
    
    for node in backup_nodes:
        if test_node(node, blockchain_type):
            logger.info(f"Switching from {current_node} to backup node: {node}")
            return node
    
    raise Exception(f"No working backup {blockchain_type} nodes available!")

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = ['models', 'reports']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

# Add this function after ensure_directories()
def load_or_create_model(model_path, model_class, X_train, y_train):
    """Load existing model or create new one if it doesn't exist."""
    try:
        if os.path.exists(model_path):
            model = model_class()
            model.load_model(model_path)
            logger.info(f"Loaded existing model from: {model_path}")
            
            # Update model with new data
            model.fit(X_train, y_train, xgb_model=model_path)
            logger.info(f"Updated model with new training data")
        else:
            model = model_class()
            model.fit(X_train, y_train)
            logger.info(f"Created new model as {model_path} did not exist")
        
        return model
    except Exception as e:
        logger.error(f"Error loading/creating model: {e}")
        # Fallback to creating new model
        model = model_class()
        model.fit(X_train, y_train)
        logger.info("Created new model due to error loading existing one")
        return model

def process_data_for_mode(df, mode, clf_model=None, reg_model=None):
    """Process data based on operation mode."""
    if mode == "TRAINING":
        # Training mode - use part of data for training, part for testing
        X = df[['vote_delay', 'author_avg_efficiency', 'author_reputation', 'author_avg_payout']]
        y_clf = df['success']
        y_reg = df['like_efficiency']

        X_train, X_test, y_clf_train, y_clf_test = train_test_split(
            X, y_clf, test_size=TEST_SIZE, random_state=42
        )
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X, y_reg, test_size=TEST_SIZE, random_state=42
        )

        # Train and save models
        clf_model = train_classifier_model(X_train, y_clf_train, X_test, y_clf_test)
        reg_model = train_regressor_model(X_reg_train, y_reg_train)

        # Generate performance reports
        generate_performance_reports(df, X_test, y_clf_test, clf_model, reg_model)

    elif mode == "TESTING":
        # Testing mode - use all data for testing existing models
        if clf_model is None or reg_model is None:
            raise ValueError("Models must be provided for testing mode")
        
        X = df[['vote_delay', 'author_avg_efficiency', 'author_reputation', 'author_avg_payout']]
        y_clf = df['success']
        
        # Generate performance reports using all data
        generate_performance_reports(df, X, y_clf, clf_model, reg_model)

    elif mode == "PRODUCTION":
        # Production mode - only make predictions, no performance evaluation
        if clf_model is None or reg_model is None:
            raise ValueError("Models must be provided for production mode")
        
        X = df[['vote_delay', 'author_avg_efficiency', 'author_reputation', 'author_avg_payout']]
        generate_predictions_report(df, X, clf_model, reg_model)

def train_classifier_model(X_train, y_train, X_test, y_test):
    """Train and evaluate classifier model."""
    model_path = os.path.join('models', 'classifier_model.json')
    clf_model = load_or_create_model(model_path, XGBClassifier, X_train, y_train)
    
    # Evaluate metrics
    y_pred = clf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    logger.info(f"Classifier Accuracy Score: {accuracy:.4f}")
    logger.info("Classification Report:\n" + class_report)
    
    # Save model
    clf_model.save_model(model_path)
    return clf_model

def train_regressor_model(X_train, y_train):
    """Train regressor model."""
    model_path = os.path.join('models', 'regressor_model.json')
    reg_model = load_or_create_model(model_path, XGBRegressor, X_train, y_train)
    reg_model.save_model(model_path)
    return reg_model

class ExcelWriter:
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

def prepare_rankings_data(author_stats):
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

def save_excel_reports(prediction_df, author_stats):
    """Save prediction results and author statistics to Excel file."""
    # Initialize Excel writer
    excel_writer = ExcelWriter('reports', CURATOR)
    
    # Prepare data dictionary for Excel sheets
    data_dict = {
        'Predictions': prediction_df,
        **prepare_rankings_data(author_stats)
    }
    
    # Save all data to Excel
    excel_writer.save_to_excel(data_dict)

def save_production_report(prediction_df):
    """Save production predictions to Excel file."""
    excel_writer = ExcelWriter('reports', CURATOR)
    
    # Prepare simplified production data
    production_data = {
        'Production Predictions': prediction_df[
            ['Post', 'Author', 'vote_decision', 
             'optimal_vote_delay_minutes', 'predicted_efficiency']
        ]
    }
    
    # Save production data
    excel_writer.save_to_excel(production_data)

def analyze_performance_results(prediction_df):
    """Analyze and log detailed performance results."""
    try:
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(prediction_df['real_success'], prediction_df['vote_decision'])
        logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
        # Calculate mean absolute error for efficiency predictions
        mae = np.mean(np.abs(prediction_df['like_efficiency'] - prediction_df['predicted_efficiency']))
        logger.info(f"Mean Absolute Error for Efficiency Predictions: {mae:.4f}")
    except Exception as e:
        logger.error(f"Error analyzing performance results: {e}")

# Modify generate_performance_reports to include performance analysis
def generate_performance_reports(df, X_test, y_test, clf_model, reg_model):
    """Generate comprehensive performance reports."""
    predictions_list = make_predictions(X_test, df, clf_model, reg_model)
    
    # Create prediction DataFrame
    prediction_df = create_prediction_dataframe(df, X_test, y_test, predictions_list)
    
    # Generate author statistics
    author_stats = generate_author_statistics(df)
    
    # Save reports
    save_excel_reports(prediction_df, author_stats)
    
    # Save aggregated author data to the database
    save_aggregated_author_data(author_stats)
    
    # Analyze and log detailed performance results
    analyze_performance_results(prediction_df)

def generate_predictions_report(df, X, clf_model, reg_model):
    """Generate production predictions report."""
    predictions_list = make_predictions(X, df, clf_model, reg_model)
    
    # Create simplified prediction DataFrame for production
    prediction_df = pd.DataFrame({
        'Post': df['Post'],
        'Author': df['Author'],
        'vote_decision': [p['vote_decision'] for p in predictions_list],
        'optimal_vote_delay_minutes': [p['optimal_vote_delay_minutes'] for p in predictions_list],
        'predicted_efficiency': [p['predicted_efficiency'] for p in predictions_list]
    })
    
    # Save production report
    save_production_report(prediction_df)

def make_predictions(X_test, df, clf_model, reg_model):
    """Make predictions using both classifier and regressor models."""
    predictions_list = []
    optimal_delay_history = df[df['success'] == 1].groupby('Author')['vote_delay'].mean().to_dict()
    
    for index, row in X_test.iterrows():
        post_features = row.to_frame().T
        author = df.loc[index, 'Author']
        vote_decision = clf_model.predict(post_features)[0]
        
        if vote_decision == 0:
            vote_decision_result = 0
            optimal_delay = None
            predicted_eff = None
        else:
            # Use historical optimal delay if available, otherwise use default
            optimal_delay = int(optimal_delay_history.get(author, 1440))  # default 24h if no history
            
            # Predict efficiency with this delay
            modified_features = post_features.copy()
            modified_features["vote_delay"] = optimal_delay
            predicted_eff = reg_model.predict(modified_features)[0]
            vote_decision_result = 1
            
        predictions_list.append({
            "vote_decision": vote_decision_result,
            "optimal_vote_delay_minutes": optimal_delay,
            "predicted_efficiency": predicted_eff
        })
    
    return predictions_list

def create_prediction_dataframe(df, X_test, y_test, predictions_list):
    """Create a DataFrame with predictions and actual values."""
    prediction_df = df.loc[X_test.index, ['Post', 'Author', 'like_efficiency', 'vote_delay']].copy()
    prediction_df = prediction_df.rename(columns={'vote_delay': 'actual_vote_delay_minutes'})
    
    # Add actual and predicted values
    prediction_df['real_success'] = y_test.values
    prediction_df['vote_decision'] = [p['vote_decision'] for p in predictions_list]
    prediction_df['optimal_vote_delay_minutes'] = [p['optimal_vote_delay_minutes'] for p in predictions_list]
    prediction_df['predicted_efficiency'] = [p['predicted_efficiency'] for p in predictions_list]
    
    # Calculate delay difference
    prediction_df['delay_difference_minutes'] = prediction_df.apply(
        lambda row: row['optimal_vote_delay_minutes'] - row['actual_vote_delay_minutes'] 
        if row['optimal_vote_delay_minutes'] is not None else None,
        axis=1
    )
    
    return prediction_df

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

def load_models():
    """Load existing models if available."""
    try:
        clf_model = XGBClassifier()
        reg_model = XGBRegressor()
        
        clf_path = os.path.join('models', 'classifier_model.json')
        reg_path = os.path.join('models', 'regressor_model.json')
        
        if os.path.exists(clf_path) and os.path.exists(reg_path):
            clf_model.load_model(clf_path)
            reg_model.load_model(reg_path)
            logger.info("Successfully loaded existing models")
            return clf_model, reg_model
        else:
            logger.error("Model files not found")
            return None, None
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None

# Ensure to close the database connection at the end of the main function
def main():
    if OPERATION_MODE not in MODE_CHOICES:
        raise ValueError(f"Invalid mode. Choose from {MODE_CHOICES}")

    logger.info(f"Starting data processing in {OPERATION_MODE} mode...")
    
    # Load models if needed
    clf_model, reg_model = None, None
    if OPERATION_MODE in ["TESTING", "PRODUCTION"]:
        clf_model, reg_model = load_models()
        if clf_model is None or reg_model is None:
            logger.error("Required models not found for testing/production mode")
            return
    
    # Initialize blockchain connection
    try:
        if BLOCKCHAIN_CHOICE == "HIVE":
            working_node = get_working_node("HIVE")
            blockchain = Hive(node=working_node)
            power_symbol = "HP"
        else:
            working_node = get_working_node("STEEM")
            blockchain = Steem(node=working_node)
            power_symbol = "SP"
            
        # Initialize the database
        author_db = AuthorDatabase()
        logger.info(f"Connected to {BLOCKCHAIN_CHOICE} node: {working_node}")
    except Exception as e:
        logger.error(f"Failed to initialize blockchain connection: {str(e)}")
        return

    # Add retry decorator for blockchain operations
    def retry_on_failure(max_retries=3, delay=1):
        def decorator(func):
            def wrapper(*args, **kwargs):
                nonlocal blockchain
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        logger.warning(f"Operation failed, retrying... ({attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        try:
                            new_node = get_working_node(BLOCKCHAIN_CHOICE)
                            blockchain = Hive(node=new_node) if BLOCKCHAIN_CHOICE == "HIVE" else Steem(node=new_node)
                            logger.info(f"Switched to new node: {new_node}")
                        except Exception as e:
                            logger.warning(f"Failed to switch node: {str(e)}")
            return wrapper
        return decorator

    @retry_on_failure()
    def get_post_data(post_identifier):
        return Comment(post_identifier, blockchain_instance=blockchain)

    @retry_on_failure()
    def get_vote_data(vote_identifier):
        return Vote(vote_identifier, blockchain_instance=blockchain)

    # Initialize data collection
    account = Account(CURATOR, blockchain_instance=blockchain)
    author_efficiency_dict = {}
    author_payout_dict = {}
    data = {
        'voting_power': [], 'vote_delay': [], 'reward': [],
        'efficiency': [], 'author_avg_efficiency': [], 'success': [],
        'author_reputation': [], 'Post': [], 'Author': [],
        'like_efficiency': [], 'author_avg_payout': []
    }

    # Collect historical data
    count = 0
    for h in account.history_reverse():
        if h['type'] == 'curation_reward':
            try:
                # Extract post info
                author = h.get('comment_author') or h.get('author')
                permlink = h.get('comment_permlink') or h.get('permlink')
                post_identifier = f"@{author}/{permlink}"
                post = get_post_data(post_identifier)

                # Get post data
                post_data = collect_post_data(
                    post, h, author, post_identifier, CURATOR,
                    blockchain, get_vote_data, author_efficiency_dict,
                    author_payout_dict, power_symbol
                )
                
                # Update data dictionary
                for key, value in post_data.items():
                    data[key].append(value)

                count += 1
                if count >= MAX_RESULTS:
                    break

            except Exception as e:
                logger.error(f"Error processing {post_identifier}: {str(e)}")
                continue

    logger.info("Data collection completed. Starting model processing...")

    # Create DataFrame and process based on operation mode
    df = pd.DataFrame(data)
    ensure_directories()
    process_data_for_mode(df, OPERATION_MODE, clf_model, reg_model)
    
    logger.info("Operation completed successfully.")
    
    # Close the database connection
    author_db.close()

def collect_post_data(post, history, author, post_identifier, curator, blockchain, 
                     get_vote_data, author_efficiency_dict, author_payout_dict, power_symbol):
    """Collect all necessary data for a single post."""
    # Get basic post info
    author_reputation = post['author_reputation']
    author_payout_token_dollar = float(str(post['author_payout_value']).split()[0])
    avg_payout = update_payout_average(author, author_payout_token_dollar, author_payout_dict)

    # Calculate times
    op_time = datetime.strptime(history['timestamp'], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
    post_creation_time = post['created']

    # Get vote data
    reward_amount_vests = float(history['reward']['amount']) / 1e6
    reward_amount = convert_vests_to_power(reward_amount_vests, blockchain)
    
    vote_identifier = f"{post_identifier}|{curator}"
    vote = get_vote_data(vote_identifier)
    vote_time = vote.time
    vote_percent = vote['percent'] / 100
    age = (vote_time - post_creation_time).total_seconds()
    
    # Calculate weights and efficiency
    weight = vote.weight / (100 if isinstance(blockchain, Steem) else 1000000000)
    teoric_reward = convert_vests_to_power(weight, blockchain)
    vote_value = teoric_reward * 2
    efficiency = (((reward_amount - teoric_reward) / teoric_reward) * 100) if vote_value > 0 else 0
    
    # Update author efficiency
    avg_efficiency = update_efficiency_average(author, efficiency, author_efficiency_dict)
    
    # Determine platform
    platform = "HIVE" if isinstance(blockchain, Hive) else "STEEM"
    
    # After calculating efficiency and before returning
    try:
        # Add post to history
        author_db.add_post_history(
            author=author,
            platform=platform,
            post_id=post_identifier,
            efficiency=efficiency,
            payout=author_payout_token_dollar
        )
    except Exception as e:
        logger.error(f"Failed to save post history: {e}")
    
    return {
        'voting_power': vote_percent,
        'vote_delay': age / 60,  # minutes
        'reward': reward_amount,
        'efficiency': efficiency,
        'author_avg_efficiency': avg_efficiency,
        'success': 1 if efficiency > 50 else 0,
        'author_reputation': author_reputation,
        'Post': post_identifier,
        'Author': author,
        'like_efficiency': efficiency,
        'author_avg_payout': avg_payout
    }

def save_aggregated_author_data(author_stats):
    """Save aggregated author data to the database."""
    try:
        platform = BLOCKCHAIN_CHOICE
        for _, row in author_stats.iterrows():
            author_db.update_author_stats(
                author=row['Author'],
                platform=platform,
                reputation=row['Reputation'],
                efficiency=row['Avg_Efficiency'],
                payout=row['Avg_Payout']
            )
        logger.info(f"Successfully updated {platform} author statistics in database")
    except Exception as e:
        logger.error(f"Error saving author data to database: {e}")

def get_author_historical_data(author, platform=None):
    """Retrieve author's historical data from database."""
    try:
        if platform:
            return author_db.get_author_stats(author, platform)
        else:
            return author_db.get_author_cross_platform_stats(author)
    except Exception as e:
        logger.error(f"Error retrieving author stats from database: {e}")
        return None

if __name__ == "__main__":
    try:
        author_db = AuthorDatabase()
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if 'author_db' in locals():
            author_db.close()
            logger.info("Database connection closed")