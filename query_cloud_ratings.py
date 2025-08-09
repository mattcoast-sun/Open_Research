#!/usr/bin/env python3
"""
Sample queries for the cloud ratings database

This script demonstrates how to query the cloud ratings database
created by create_cloud_ratings_database.py
"""

import sqlite3
import os
from typing import List, Tuple


def connect_to_database() -> sqlite3.Connection:
    """Connect to the cloud ratings database."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'cloud_ratings.db')
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        print("Please run create_cloud_ratings_database.py first")
        exit(1)
    
    return sqlite3.connect(db_path)


def print_table_header(columns: List[str], widths: List[int]) -> None:
    """Print a formatted table header."""
    header = " | ".join(col.ljust(width) for col, width in zip(columns, widths))
    print(header)
    print("-" * len(header))


def query_top_providers(conn: sqlite3.Connection, limit: int = 5) -> None:
    """Query and display top cloud providers by aggregate score."""
    print(f"\n🏆 TOP {limit} CLOUD PROVIDERS BY AGGREGATE SCORE")
    print("=" * 60)
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT provider, aggregate_score, ranking 
        FROM cloud_ratings_summary 
        LIMIT ?
    """, (limit,))
    
    results = cursor.fetchall()
    print_table_header(["Rank", "Provider", "Score"], [4, 20, 8])
    
    for row in results:
        print(f"{row[2]:>4} | {row[0]:<20} | {row[1]:>8.2f}")


def query_ai_leaders(conn: sqlite3.Connection, threshold: float = 9.0) -> None:
    """Query providers with high AI capabilities."""
    print(f"\n🤖 PROVIDERS WITH AI CAPABILITIES > {threshold}")
    print("=" * 50)
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT provider, ai_capabilities, aggregate_score
        FROM cloud_ratings 
        WHERE ai_capabilities > ?
        ORDER BY ai_capabilities DESC
    """, (threshold,))
    
    results = cursor.fetchall()
    
    if results:
        print_table_header(["Provider", "AI Score", "Overall Score"], [20, 10, 13])
        for row in results:
            print(f"{row[0]:<20} | {row[1]:>10.1f} | {row[2]:>13.2f}")
    else:
        print(f"No providers found with AI capabilities > {threshold}")


def query_cost_efficient(conn: sqlite3.Connection, threshold: float = 8.0) -> None:
    """Query most cost-efficient providers."""
    print(f"\n💰 MOST COST-EFFICIENT PROVIDERS (Cost Efficiency > {threshold})")
    print("=" * 60)
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT provider, cost_efficiency, aggregate_score
        FROM cloud_ratings 
        WHERE cost_efficiency > ?
        ORDER BY cost_efficiency DESC
    """, (threshold,))
    
    results = cursor.fetchall()
    
    if results:
        print_table_header(["Provider", "Cost Efficiency", "Overall Score"], [20, 15, 13])
        for row in results:
            print(f"{row[0]:<20} | {row[1]:>15.1f} | {row[2]:>13.2f}")
    else:
        print(f"No providers found with cost efficiency > {threshold}")


def query_sustainability_leaders(conn: sqlite3.Connection) -> None:
    """Query providers by sustainability score."""
    print(f"\n🌱 SUSTAINABILITY LEADERS")
    print("=" * 40)
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT provider, sustainability_score, aggregate_score
        FROM cloud_ratings 
        ORDER BY sustainability_score DESC
        LIMIT 5
    """)
    
    results = cursor.fetchall()
    print_table_header(["Provider", "Sustainability", "Overall Score"], [20, 13, 13])
    
    for row in results:
        print(f"{row[0]:<20} | {row[1]:>13.1f} | {row[2]:>13.2f}")


def query_provider_details(conn: sqlite3.Connection, provider_name: str) -> None:
    """Query detailed information for a specific provider."""
    print(f"\n📊 DETAILED METRICS FOR {provider_name.upper()}")
    print("=" * 50)
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM cloud_ratings WHERE provider = ?
    """, (provider_name,))
    
    result = cursor.fetchone()
    
    if result:
        columns = [description[0] for description in cursor.description]
        
        # Skip id, created_at, updated_at for cleaner display
        skip_columns = {'id', 'created_at', 'updated_at'}
        
        for i, (col, value) in enumerate(zip(columns, result)):
            if col not in skip_columns:
                if isinstance(value, float):
                    print(f"{col.replace('_', ' ').title():<25}: {value:>6.1f}")
                else:
                    print(f"{col.replace('_', ' ').title():<25}: {value}")
    else:
        print(f"Provider '{provider_name}' not found in database")


def query_comparison(conn: sqlite3.Connection, providers: List[str]) -> None:
    """Compare multiple providers side by side."""
    print(f"\n⚖️  PROVIDER COMPARISON")
    print("=" * 80)
    
    cursor = conn.cursor()
    placeholders = ','.join(['?' for _ in providers])
    cursor.execute(f"""
        SELECT provider, ai_capabilities, performance, cost_efficiency, 
               sustainability_score, aggregate_score
        FROM cloud_ratings 
        WHERE provider IN ({placeholders})
        ORDER BY aggregate_score DESC
    """, providers)
    
    results = cursor.fetchall()
    
    if results:
        print_table_header(
            ["Provider", "AI", "Perf", "Cost", "Sustain", "Overall"], 
            [15, 5, 5, 5, 7, 8]
        )
        
        for row in results:
            print(f"{row[0]:<15} | {row[1]:>5.1f} | {row[2]:>5.1f} | {row[3]:>5.1f} | {row[4]:>7.1f} | {row[5]:>8.2f}")
    else:
        print("No matching providers found")


def main():
    """Main function to demonstrate database queries."""
    print("Cloud Ratings Database Query Examples")
    print("=" * 50)
    
    try:
        conn = connect_to_database()
        
        # Various query examples
        query_top_providers(conn, 5)
        query_ai_leaders(conn, 9.0)
        query_cost_efficient(conn, 8.0)
        query_sustainability_leaders(conn)
        query_provider_details(conn, "AWS")
        query_comparison(conn, ["AWS", "GCP", "Azure"])
        
        print(f"\n✓ Query examples completed successfully!")
        print("\nYou can modify this script to run your own custom queries.")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    main()
