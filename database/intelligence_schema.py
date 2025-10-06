#!/usr/bin/env python3
"""
Business Dealer Intelligence Database Schema
Creates and manages intelligence system tables for user behavior tracking,
opportunity scoring, and recommendation management.

Following patterns from existing database setup scripts.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import json


class IntelligenceSchema:
    """Intelligence database schema manager"""

    def __init__(self, db_path: str = "intelligence.db"):
        self.db_path = db_path
        self.base_path = os.path.dirname(db_path) or "."

    def get_schema_sql(self) -> str:
        """Get the complete SQL schema for intelligence tables"""
        return """
        -- User Interactions Table
        -- Tracks all user interactions with opportunities for behavior analysis
        CREATE TABLE IF NOT EXISTS user_interactions (
            id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
            user_id TEXT NOT NULL,
            opportunity_id TEXT NOT NULL,
            interaction_type TEXT NOT NULL CHECK (interaction_type IN ('view', 'click', 'share', 'contact', 'save')),
            timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            duration INTEGER,  -- Duration in seconds
            metadata TEXT,     -- JSON metadata
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- Opportunity Scores Table
        -- Stores calculated relevance and success probability scores
        CREATE TABLE IF NOT EXISTS opportunity_scores (
            id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
            opportunity_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            relevance_score REAL NOT NULL CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0),
            success_probability REAL NOT NULL CHECK (success_probability >= 0.0 AND success_probability <= 1.0),
            factors TEXT,      -- JSON factors that influenced the score
            calculated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- Recommendations Table
        -- Stores generated recommendations for users
        CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
            user_id TEXT NOT NULL,
            opportunity_id TEXT NOT NULL,
            recommendation_type TEXT NOT NULL,
            score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
            reasoning TEXT,
            generated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            clicked_at DATETIME,
            dismissed_at DATETIME,
            expires_at DATETIME,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- User Behavior Metrics Table
        -- Aggregated metrics for user behavior analysis
        CREATE TABLE IF NOT EXISTS user_behavior_metrics (
            id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
            user_id TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            value REAL NOT NULL,
            period TEXT NOT NULL,  -- 'daily', 'weekly', 'monthly'
            period_start DATETIME NOT NULL,
            period_end DATETIME NOT NULL,
            calculated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,         -- JSON metadata
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- Network Analytics Table
        -- Tracks connection strength and interaction patterns
        CREATE TABLE IF NOT EXISTS network_analytics (
            id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
            user_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            connection_strength REAL NOT NULL CHECK (connection_strength >= 0.0 AND connection_strength <= 1.0),
            interaction_frequency REAL NOT NULL DEFAULT 0.0,
            last_interaction DATETIME,
            total_interactions INTEGER NOT NULL DEFAULT 0,
            successful_connections INTEGER NOT NULL DEFAULT 0,
            metadata TEXT,         -- JSON metadata
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- Embedding Cache Table
        -- Stores vector embeddings for opportunities and content
        CREATE TABLE IF NOT EXISTS embedding_cache (
            id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
            content_id TEXT NOT NULL,
            content_type TEXT NOT NULL,  -- 'opportunity', 'user_profile', 'content'
            embedding_model TEXT NOT NULL,
            embedding_vector TEXT NOT NULL,  -- JSON array of floats
            content_hash TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME
        );

        -- Notification History Table
        -- Tracks sent notifications and their effectiveness
        CREATE TABLE IF NOT EXISTS notification_history (
            id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
            user_id TEXT NOT NULL,
            notification_type TEXT NOT NULL,
            content TEXT NOT NULL,
            priority_score REAL NOT NULL CHECK (priority_score >= 0.0 AND priority_score <= 1.0),
            sent_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            opened_at DATETIME,
            clicked_at DATETIME,
            dismissed_at DATETIME,
            delivery_channel TEXT NOT NULL,  -- 'email', 'push', 'in_app'
            metadata TEXT,         -- JSON metadata
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        -- Performance Indexes for optimal query performance
        -- User Interactions indexes
        CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_interactions_opportunity_id ON user_interactions(opportunity_id);
        CREATE INDEX IF NOT EXISTS idx_user_interactions_timestamp ON user_interactions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_user_interactions_type ON user_interactions(interaction_type);
        CREATE INDEX IF NOT EXISTS idx_user_interaction_time ON user_interactions(user_id, timestamp);

        -- Opportunity Scores indexes
        CREATE INDEX IF NOT EXISTS idx_opportunity_scores_user_id ON opportunity_scores(user_id);
        CREATE INDEX IF NOT EXISTS idx_opportunity_scores_opportunity_id ON opportunity_scores(opportunity_id);
        CREATE INDEX IF NOT EXISTS idx_opportunity_scores_calculated_at ON opportunity_scores(calculated_at);
        CREATE INDEX IF NOT EXISTS idx_opportunity_scores_relevance ON opportunity_scores(relevance_score);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_user_opportunity_score ON opportunity_scores(user_id, opportunity_id);

        -- Recommendations indexes
        CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id);
        CREATE INDEX IF NOT EXISTS idx_recommendations_opportunity_id ON recommendations(opportunity_id);
        CREATE INDEX IF NOT EXISTS idx_recommendations_generated_at ON recommendations(generated_at);
        CREATE INDEX IF NOT EXISTS idx_recommendations_score ON recommendations(score);
        CREATE INDEX IF NOT EXISTS idx_recommendations_type ON recommendations(recommendation_type);

        -- User Behavior Metrics indexes
        CREATE INDEX IF NOT EXISTS idx_behavior_metrics_user_id ON user_behavior_metrics(user_id);
        CREATE INDEX IF NOT EXISTS idx_behavior_metrics_type ON user_behavior_metrics(metric_type);
        CREATE INDEX IF NOT EXISTS idx_behavior_metrics_period ON user_behavior_metrics(period);
        CREATE INDEX IF NOT EXISTS idx_behavior_metrics_period_start ON user_behavior_metrics(period_start);

        -- Network Analytics indexes
        CREATE INDEX IF NOT EXISTS idx_network_analytics_user_id ON network_analytics(user_id);
        CREATE INDEX IF NOT EXISTS idx_network_analytics_contact_id ON network_analytics(contact_id);
        CREATE INDEX IF NOT EXISTS idx_network_analytics_strength ON network_analytics(connection_strength);
        CREATE INDEX IF NOT EXISTS idx_network_analytics_last_interaction ON network_analytics(last_interaction);

        -- Embedding Cache indexes
        CREATE INDEX IF NOT EXISTS idx_embedding_cache_content_id ON embedding_cache(content_id);
        CREATE INDEX IF NOT EXISTS idx_embedding_cache_content_type ON embedding_cache(content_type);
        CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(embedding_model);
        CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash ON embedding_cache(content_hash);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_embedding_cache_unique ON embedding_cache(content_id, content_type, embedding_model);

        -- Notification History indexes
        CREATE INDEX IF NOT EXISTS idx_notification_history_user_id ON notification_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_notification_history_sent_at ON notification_history(sent_at);
        CREATE INDEX IF NOT EXISTS idx_notification_history_type ON notification_history(notification_type);
        CREATE INDEX IF NOT EXISTS idx_notification_history_priority ON notification_history(priority_score);
        """

    def create_database(self, db_path: Optional[str] = None) -> bool:
        """
        Create intelligence database with all tables and indexes

        Args:
            db_path: Optional path to database file

        Returns:
            bool: True if successful
        """
        if db_path is None:
            db_path = self.db_path

        try:
            print(f"🧠 Creating intelligence database: {db_path}")

            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # Remove existing database if it exists
            if os.path.exists(db_path):
                os.remove(db_path)

            # Create database and execute schema
            conn = sqlite3.connect(db_path)
            conn.executescript(self.get_schema_sql())
            conn.close()

            # Verify creation
            file_size = os.path.getsize(db_path)
            print(f"✅ Intelligence database created: {db_path} ({file_size:,} bytes)")

            return True

        except Exception as e:
            print(f"❌ Error creating intelligence database {db_path}: {e}")
            return False

    def seed_sample_data(self, db_path: Optional[str] = None) -> bool:
        """
        Seed database with sample data for testing

        Args:
            db_path: Optional path to database file

        Returns:
            bool: True if successful
        """
        if db_path is None:
            db_path = self.db_path

        try:
            print(f"🌱 Seeding sample data into: {db_path}")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Generate sample user interactions
            sample_users = ["user1", "user2", "user3", "user4", "user5"]
            sample_opportunities = ["opp1", "opp2", "opp3", "opp4", "opp5"]
            interaction_types = ["view", "click", "share", "contact", "save"]

            # Insert sample user interactions
            for i in range(100):
                user_id = f"user_{i % 5 + 1}"
                opportunity_id = f"opp_{i % 5 + 1}"
                interaction_type = interaction_types[i % len(interaction_types)]
                timestamp = datetime.now() - timedelta(days=i % 30)
                duration = 30 + (i % 300)  # 30-330 seconds

                cursor.execute(
                    """
                    INSERT INTO user_interactions (user_id, opportunity_id, interaction_type, timestamp, duration, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        opportunity_id,
                        interaction_type,
                        timestamp,
                        duration,
                        json.dumps({"session_id": f"session_{i}", "device": "web"}),
                    ),
                )

            # Insert sample opportunity scores
            for user_id in sample_users:
                for opportunity_id in sample_opportunities:
                    relevance_score = min(
                        1.0,
                        max(0.0, 0.5 + (hash(user_id + opportunity_id) % 100) / 200),
                    )
                    success_probability = min(
                        1.0,
                        max(0.0, 0.3 + (hash(opportunity_id + user_id) % 100) / 200),
                    )

                    cursor.execute(
                        """
                        INSERT INTO opportunity_scores 
                        (opportunity_id, user_id, relevance_score, success_probability, factors, calculated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            opportunity_id,
                            user_id,
                            relevance_score,
                            success_probability,
                            json.dumps(
                                {
                                    "industry_match": 0.8,
                                    "location_match": 0.6,
                                    "behavior_match": 0.7,
                                }
                            ),
                            datetime.now(),
                        ),
                    )

            # Insert sample recommendations
            for i in range(50):
                user_id = sample_users[i % len(sample_users)]
                opportunity_id = sample_opportunities[i % len(sample_opportunities)]
                recommendation_type = [
                    "personalized",
                    "trending",
                    "similar_users",
                    "location_based",
                ][i % 4]
                score = min(1.0, max(0.0, 0.6 + (i % 40) / 100))

                cursor.execute(
                    """
                    INSERT INTO recommendations 
                    (user_id, opportunity_id, recommendation_type, score, reasoning, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        opportunity_id,
                        recommendation_type,
                        score,
                        f"Recommended based on {recommendation_type} analysis",
                        datetime.now() - timedelta(hours=i % 48),
                    ),
                )

            # Insert sample behavior metrics
            for user_id in sample_users:
                for metric_type in [
                    "engagement_rate",
                    "click_through_rate",
                    "conversion_rate",
                ]:
                    value = max(
                        0.0, min(1.0, 0.1 + (hash(user_id + metric_type) % 80) / 100)
                    )
                    period_start = datetime.now() - timedelta(days=7)
                    period_end = datetime.now()

                    cursor.execute(
                        """
                        INSERT INTO user_behavior_metrics 
                        (user_id, metric_type, value, period, period_start, period_end, calculated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            user_id,
                            metric_type,
                            value,
                            "weekly",
                            period_start,
                            period_end,
                            datetime.now(),
                        ),
                    )

            conn.commit()
            conn.close()

            print("✅ Sample data seeded successfully")
            return True

        except Exception as e:
            print(f"❌ Error seeding sample data: {e}")
            return False

    def verify_database(self, db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify database structure and data

        Args:
            db_path: Optional path to database file

        Returns:
            Dict: Verification results
        """
        if db_path is None:
            db_path = self.db_path

        results = {
            "database_exists": False,
            "tables": {},
            "indexes": {},
            "sample_data": {},
        }

        try:
            if not os.path.exists(db_path):
                results["error"] = f"Database file not found: {db_path}"
                return results

            results["database_exists"] = True
            results["file_size"] = os.path.getsize(db_path)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                results["tables"][table] = count

            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            results["indexes"] = indexes

            # Sample data from key tables
            if "user_interactions" in tables:
                cursor.execute(
                    "SELECT interaction_type, COUNT(*) FROM user_interactions GROUP BY interaction_type"
                )
                results["sample_data"]["interaction_types"] = dict(cursor.fetchall())

            if "opportunity_scores" in tables:
                cursor.execute(
                    "SELECT AVG(relevance_score), AVG(success_probability) FROM opportunity_scores"
                )
                avg_scores = cursor.fetchone()
                results["sample_data"]["avg_scores"] = {
                    "relevance": avg_scores[0],
                    "success_probability": avg_scores[1],
                }

            conn.close()

            print("✅ Database verification completed")

        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Error verifying database: {e}")

        return results


def main():
    """Main function for database setup"""
    print("🚀 BUSINESS DEALER INTELLIGENCE DATABASE SETUP")
    print("=" * 60)

    # Create schema manager
    schema_manager = IntelligenceSchema("database/intelligence.db")

    # Create database
    success = schema_manager.create_database()

    if success:
        # Seed sample data
        schema_manager.seed_sample_data()

        # Verify database
        results = schema_manager.verify_database()

        print("\n📊 DATABASE VERIFICATION RESULTS")
        print("=" * 50)
        print(f"Database exists: {results['database_exists']}")
        print(f"File size: {results.get('file_size', 0):,} bytes")
        print(f"Tables created: {len(results['tables'])}")
        print(f"Indexes created: {len(results['indexes'])}")

        if results["tables"]:
            print("\nTable record counts:")
            for table, count in results["tables"].items():
                print(f"  {table}: {count:,} records")

        if results.get("sample_data"):
            print("\nSample data summary:")
            for key, value in results["sample_data"].items():
                print(f"  {key}: {value}")

        print("\n🎉 Intelligence database setup completed successfully!")
        return True
    else:
        print("\n❌ Database setup failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
