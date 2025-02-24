import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path="database/author_stats.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create authors table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS authors (
                    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author_name TEXT UNIQUE NOT NULL
                )
                ''')

                # Create author statistics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS author_statistics (
                    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author_id INTEGER,
                    avg_efficiency REAL,
                    reputation REAL,
                    avg_payout REAL,
                    training_date TIMESTAMP,
                    model_version TEXT,
                    FOREIGN KEY (author_id) REFERENCES authors(author_id)
                )
                ''')

                # Create aggregated statistics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS aggregated_statistics (
                    author_id INTEGER,
                    avg_efficiency_all_time REAL,
                    reputation_all_time REAL,
                    avg_payout_all_time REAL,
                    total_trainings INTEGER,
                    last_updated TIMESTAMP,
                    FOREIGN KEY (author_id) REFERENCES authors(author_id),
                    UNIQUE(author_id)
                )
                ''')

                conn.commit()
                logger.info("Database initialized successfully")

        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def update_author_stats(self, author_name, efficiency, reputation, payout, model_version):
        """Update author statistics in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert or get author
                cursor.execute('''
                INSERT OR IGNORE INTO authors (author_name)
                VALUES (?)
                ''', (author_name,))
                
                cursor.execute('SELECT author_id FROM authors WHERE author_name = ?', 
                             (author_name,))
                author_id = cursor.fetchone()[0]

                # Insert new statistics
                cursor.execute('''
                INSERT INTO author_statistics 
                (author_id, avg_efficiency, reputation, avg_payout, training_date, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (author_id, efficiency, reputation, payout, 
                     datetime.now(), model_version))

                # Update aggregated statistics
                cursor.execute('''
                INSERT INTO aggregated_statistics 
                (author_id, avg_efficiency_all_time, reputation_all_time, 
                 avg_payout_all_time, total_trainings, last_updated)
                SELECT 
                    author_id,
                    AVG(avg_efficiency),
                    AVG(reputation),
                    AVG(avg_payout),
                    COUNT(*),
                    CURRENT_TIMESTAMP
                FROM author_statistics
                WHERE author_id = ?
                GROUP BY author_id
                ON CONFLICT(author_id) DO UPDATE SET
                    avg_efficiency_all_time = excluded.avg_efficiency_all_time,
                    reputation_all_time = excluded.reputation_all_time,
                    avg_payout_all_time = excluded.avg_payout_all_time,
                    total_trainings = excluded.total_trainings,
                    last_updated = CURRENT_TIMESTAMP
                ''', (author_id,))

                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Error updating author stats: {e}")
            raise

    def get_author_stats(self, author_name):
        """Retrieve author statistics from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT 
                    a.author_name,
                    ag.avg_efficiency_all_time,
                    ag.reputation_all_time,
                    ag.avg_payout_all_time,
                    ag.total_trainings,
                    ag.last_updated
                FROM authors a
                JOIN aggregated_statistics ag ON a.author_id = ag.author_id
                WHERE a.author_name = ?
                ''', (author_name,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'author_name': result[0],
                        'avg_efficiency': result[1],
                        'reputation': result[2],
                        'avg_payout': result[3],
                        'total_trainings': result[4],
                        'last_updated': result[5]
                    }
                return None

        except sqlite3.Error as e:
            logger.error(f"Error retrieving author stats: {e}")
            raise