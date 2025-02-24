import sqlite3
from datetime import datetime
import logging

class AuthorDatabase:
    def __init__(self, db_path="author_stats.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
            raise

    def create_tables(self):
        """Create required database tables if they don't exist."""
        try:
            self.cursor.executescript('''
                CREATE TABLE IF NOT EXISTS authors (
                    author TEXT,
                    platform TEXT,  -- 'HIVE' or 'STEEM'
                    reputation REAL,
                    avg_efficiency REAL,
                    avg_payout REAL,
                    total_posts INTEGER,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (author, platform)
                );

                CREATE TABLE IF NOT EXISTS author_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author TEXT,
                    platform TEXT,
                    post_id TEXT,
                    efficiency REAL,
                    payout REAL,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (author, platform) REFERENCES authors(author, platform)
                );
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error creating tables: {e}")
            raise

    def update_author_stats(self, author, platform, reputation, efficiency, payout):
        """Update author statistics in database."""
        try:
            # Get current stats
            self.cursor.execute('''
                SELECT avg_efficiency, avg_payout, total_posts 
                FROM authors WHERE author = ? AND platform = ?
            ''', (author, platform))
            result = self.cursor.fetchone()

            if result:
                # Update existing author
                old_avg_eff, old_avg_payout, total_posts = result
                total_posts += 1
                new_avg_eff = ((old_avg_eff * (total_posts-1)) + efficiency) / total_posts
                new_avg_payout = ((old_avg_payout * (total_posts-1)) + payout) / total_posts

                self.cursor.execute('''
                    UPDATE authors 
                    SET reputation = ?,
                        avg_efficiency = ?,
                        avg_payout = ?,
                        total_posts = ?,
                        last_updated = ?
                    WHERE author = ? AND platform = ?
                ''', (reputation, new_avg_eff, new_avg_payout, total_posts, 
                      datetime.now(), author, platform))
            else:
                # Insert new author
                self.cursor.execute('''
                    INSERT INTO authors 
                    (author, platform, reputation, avg_efficiency, avg_payout, total_posts, last_updated)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                ''', (author, platform, reputation, efficiency, payout, datetime.now()))

            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error updating author stats: {e}")
            self.conn.rollback()
            raise

    def add_post_history(self, author, platform, post_id, efficiency, payout):
        """Add new post to author history."""
        try:
            self.cursor.execute('''
                INSERT INTO author_history
                (author, platform, post_id, efficiency, payout, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (author, platform, post_id, efficiency, payout, datetime.now()))
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error adding post history: {e}") 
            self.conn.rollback()
            raise

    def get_author_stats(self, author, platform):
        """Retrieve author statistics for specific platform."""
        try:
            self.cursor.execute('''
                SELECT reputation, avg_efficiency, avg_payout, total_posts
                FROM authors WHERE author = ? AND platform = ?
            ''', (author, platform))
            result = self.cursor.fetchone()
            if result:
                return {
                    'reputation': result[0],
                    'avg_efficiency': result[1], 
                    'avg_payout': result[2],
                    'total_posts': result[3],
                    'platform': platform
                }
            return None
        except sqlite3.Error as e:
            logging.error(f"Error getting author stats: {e}")
            raise

    def get_author_cross_platform_stats(self, author):
        """Retrieve author statistics across all platforms."""
        try:
            self.cursor.execute('''
                SELECT platform, reputation, avg_efficiency, avg_payout, total_posts
                FROM authors WHERE author = ?
            ''', (author,))
            results = self.cursor.fetchall()
            if results:
                return {row[0]: {
                    'reputation': row[1],
                    'avg_efficiency': row[2],
                    'avg_payout': row[3],
                    'total_posts': row[4]
                } for row in results}
            return None
        except sqlite3.Error as e:
            logging.error(f"Error getting cross-platform stats: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()