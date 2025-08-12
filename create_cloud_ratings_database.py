#!/usr/bin/env python3
"""
Script to create a SQL database from cloud_ratings_benchmarks.json

This script creates a SQLite database with a table structure that matches
the cloud ratings benchmarks data and imports all records from the JSON file.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any


def create_database_connection(db_path: str) -> sqlite3.Connection:
    """Create and return a database connection."""
    try:
        conn = sqlite3.connect(db_path)
        print(f"✓ Successfully connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"✗ Error connecting to database: {e}")
        raise


def create_cloud_ratings_table(conn: sqlite3.Connection) -> None:
    """Create the cloud_ratings table with appropriate schema."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS cloud_ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT NOT NULL UNIQUE,
        ai_capabilities REAL NOT NULL,
        performance REAL NOT NULL,
        cost_efficiency REAL NOT NULL,
        flexibility REAL NOT NULL,
        customer_service REAL NOT NULL,
        sustainability_score REAL NOT NULL,
        ecosystem_innovation REAL NOT NULL,
        data_sovereignty_strength REAL NOT NULL,
        aggregate_score REAL NOT NULL,
        scoring_version TEXT NOT NULL,
        last_updated DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        print("✓ Successfully created cloud_ratings table")
    except sqlite3.Error as e:
        print(f"✗ Error creating table: {e}")
        raise


def create_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes for better query performance."""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_provider ON cloud_ratings(provider);",
        "CREATE INDEX IF NOT EXISTS idx_aggregate_score ON cloud_ratings(aggregate_score);",
        "CREATE INDEX IF NOT EXISTS idx_scoring_version ON cloud_ratings(scoring_version);",
        "CREATE INDEX IF NOT EXISTS idx_last_updated ON cloud_ratings(last_updated);"
    ]
    
    try:
        cursor = conn.cursor()
        for index_sql in indexes:
            cursor.execute(index_sql)
        conn.commit()
        print("✓ Successfully created database indexes")
    except sqlite3.Error as e:
        print(f"✗ Error creating indexes: {e}")
        raise


def load_json_data(json_file_path: str) -> List[Dict[str, Any]]:
    """Load and return data from JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"✓ Successfully loaded {len(data)} records from {json_file_path}")
        return data
    except FileNotFoundError:
        print(f"✗ Error: JSON file not found: {json_file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"✗ Error decoding JSON: {e}")
        raise


def insert_cloud_ratings_data(conn: sqlite3.Connection, data: List[Dict[str, Any]]) -> None:
    """Insert cloud ratings data into the database."""
    insert_sql = """
    INSERT OR REPLACE INTO cloud_ratings (
        provider, ai_capabilities, performance, cost_efficiency, flexibility,
        customer_service, sustainability_score, ecosystem_innovation,
        data_sovereignty_strength, aggregate_score, scoring_version, last_updated
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        cursor = conn.cursor()
        
        for record in data:
            cursor.execute(insert_sql, (
                record['provider'],
                record['ai_capabilities'],
                record['performance'],
                record['cost_efficiency'],
                record['flexibility'],
                record['customer_service'],
                record['sustainability_score'],
                record['ecosystem_innovation'],
                record['data_sovereignty_strength'],
                record['aggregate_score'],
                record['scoring_version'],
                record['last_updated']
            ))
        
        conn.commit()
        print(f"✓ Successfully inserted {len(data)} records into cloud_ratings table")
        
    except sqlite3.Error as e:
        print(f"✗ Error inserting data: {e}")
        raise


def create_summary_view(conn: sqlite3.Connection) -> None:
    """Create a summary view for easy data analysis."""
    view_sql = """
    CREATE VIEW IF NOT EXISTS cloud_ratings_summary AS
    SELECT 
        provider,
        aggregate_score,
        RANK() OVER (ORDER BY aggregate_score DESC) as ranking,
        ai_capabilities,
        performance,
        cost_efficiency,
        flexibility,
        customer_service,
        sustainability_score,
        ecosystem_innovation,
        data_sovereignty_strength,
        scoring_version,
        last_updated
    FROM cloud_ratings
    ORDER BY aggregate_score DESC;
    """
    
    try:
        cursor = conn.cursor()
        cursor.execute(view_sql)
        conn.commit()
        print("✓ Successfully created cloud_ratings_summary view")
    except sqlite3.Error as e:
        print(f"✗ Error creating view: {e}")
        raise


def verify_data(conn: sqlite3.Connection) -> None:
    """Verify the imported data by running some basic queries."""
    print("\n" + "="*50)
    print("DATABASE VERIFICATION")
    print("="*50)
    
    try:
        cursor = conn.cursor()
        
        # Count total records
        cursor.execute("SELECT COUNT(*) FROM cloud_ratings")
        count = cursor.fetchone()[0]
        print(f"Total records in database: {count}")
        
        # Show top 5 providers by aggregate score
        cursor.execute("""
            SELECT provider, aggregate_score, ranking 
            FROM cloud_ratings_summary 
            LIMIT 5
        """)
        
        print("\nTop 5 Cloud Providers by Aggregate Score:")
        print("-" * 45)
        for row in cursor.fetchall():
            print(f"{row[2]:2d}. {row[0]:<15} - {row[1]:.2f}")
        
        # Show scoring version distribution
        cursor.execute("""
            SELECT scoring_version, COUNT(*) as count 
            FROM cloud_ratings 
            GROUP BY scoring_version
        """)
        
        print("\nScoring Version Distribution:")
        print("-" * 30)
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]} records")
            
    except sqlite3.Error as e:
        print(f"✗ Error during verification: {e}")


def main():
    """Main function to orchestrate the database creation process."""
    print("Cloud Ratings Database Creator")
    print("=" * 50)
    
    # Configuration - support both local dev and container environments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for JSON file in multiple locations
    json_file_candidates = [
        os.path.join(script_dir, 'cloud_ratings_benchmarks.json'),
        './cloud_ratings_benchmarks.json',
        '/app/cloud_ratings_benchmarks.json'
    ]
    
    json_file_path = None
    for candidate in json_file_candidates:
        if os.path.exists(candidate):
            json_file_path = candidate
            break
    
    if not json_file_path:
        json_file_path = json_file_candidates[0]  # Use first as default
    
    # Database path - prefer container data directory
    db_path = os.getenv("DB_PATH", "/app/data/cloud_ratings.db")
    if not os.path.exists(os.path.dirname(db_path)):
        # Fallback to script directory for local development
        db_path = os.path.join(script_dir, 'cloud_ratings.db')
    
    # Check if JSON file exists
    if not os.path.exists(json_file_path):
        print(f"✗ Error: JSON file not found at {json_file_path}")
        return
    
    try:
        # Create database connection
        conn = create_database_connection(db_path)
        
        # Create table schema
        create_cloud_ratings_table(conn)
        
        # Create indexes for performance
        create_indexes(conn)
        
        # Load JSON data
        data = load_json_data(json_file_path)
        
        # Insert data into database
        insert_cloud_ratings_data(conn, data)
        
        # Create summary view
        create_summary_view(conn)
        
        # Verify the imported data
        verify_data(conn)
        
        print(f"\n✓ Database successfully created at: {db_path}")
        print("✓ You can now query the database using any SQLite client or Python")
        
        # Example queries
        print("\nExample queries you can run:")
        print("1. SELECT * FROM cloud_ratings ORDER BY aggregate_score DESC;")
        print("2. SELECT * FROM cloud_ratings_summary;")
        print("3. SELECT provider, ai_capabilities FROM cloud_ratings WHERE ai_capabilities > 9.0;")
        
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        return
    
    finally:
        if 'conn' in locals():
            conn.close()
            print("\n✓ Database connection closed")


if __name__ == "__main__":
    main()
