import logging
from datetime import datetime, timezone
from src.settings.config import (
    BLOCKCHAIN_CHOICE,
    CURATOR,
    STEEM_NODES,
    HIVE_NODES
)
from src.settings.logger_config import logger
from src.utils.beem import (
    initialize_blockchain,
    get_post_data,
    get_vote_data,
    convert_vests_to_power
)
from src.data.data_collector import BlockchainDataCollector
from src.data.data_processor import DataProcessor
from src.models.classifier import VoteClassifier
from src.models.regressor import EfficiencyRegressor
from src.analysis.metrics import ModelMetrics
from src.analysis.reporting import PerformanceAnalyzer

def main():
    logger.info("Starting data processing...")

    try:
        # Initialize blockchain connection
        blockchain, power_symbol, working_node = initialize_blockchain(BLOCKCHAIN_CHOICE)
        logger.info(f"Connected to {BLOCKCHAIN_CHOICE} node: {working_node}")

        # Initialize data collector
        collector = BlockchainDataCollector(blockchain, CURATOR, power_symbol)
        raw_data = collector.collect_account_history(limit=1000)
        logger.info(f"Collected {len(raw_data['Post'])} historical records")

        # Process collected data
        processor = DataProcessor(raw_data)
        (X_train, X_test, y_clf_train, y_clf_test,
         X_reg_train, X_reg_test, y_reg_train, y_reg_test) = processor.prepare_training_data()
        logger.info("Data processing completed. Starting model training...")

        # Train classifier model
        classifier = VoteClassifier()
        classifier.train(X_train, y_clf_train)
        clf_predictions = classifier.predict(X_test)
        classifier.save_model('models/classifier_model.json')

        # Train regressor model
        regressor = EfficiencyRegressor()
        regressor.train(X_reg_train, y_reg_train)
        reg_predictions = regressor.predict(X_reg_test)
        regressor.save_model('models/regressor_model.json')
        logger.info("Models trained and saved successfully")

        # Calculate metrics
        metrics = ModelMetrics()
        clf_metrics = metrics.calculate_classifier_metrics(y_clf_test, clf_predictions)
        reg_metrics = metrics.calculate_regressor_metrics(y_reg_test, reg_predictions)
        metrics.log_metrics()

        # Generate predictions
        prediction_df = processor.df.loc[X_test.index].copy()
        prediction_df['vote_decision'] = clf_predictions
        prediction_df['predicted_efficiency'] = reg_predictions

        # Fix: Apply optimal delay using pandas methods
        prediction_df['optimal_vote_delay_minutes'] = prediction_df['Author'].map(
            processor.optimal_delay_history
        ).fillna(1440)  # default 24h if no history

        # Generate analysis and reports
        analyzer = PerformanceAnalyzer(prediction_df, {
            'classifier_metrics': clf_metrics,
            'regressor_metrics': reg_metrics
        })

        # Generate visualizations
        analyzer.plot_efficiency_distribution('reports/figures/efficiency_dist.png')
        analyzer.plot_delay_vs_efficiency('reports/figures/delay_efficiency.png')
        
        # Generate performance report
        analyzer.generate_performance_report('reports/performance_report.md')
        
        # Save results to Excel
        processor.save_results_to_excel(
            prediction_df,
            CURATOR,
            f'reports/predictions_and_rankings_{CURATOR}.xlsx'
        )

        logger.info("Operation completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()