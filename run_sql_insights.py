# run_sql_insights.py
import pandas as pd
import sqlite3
from datetime import datetime

class SQLHouseAnalyzer:
    def __init__(self, data_path='data/house_data_tableau_fixed.csv'):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.conn = sqlite3.connect(':memory:')
        self.df.to_sql('house_data', self.conn, index=False)
        print(f"✓ Loaded {len(self.df):,} properties from {data_path}")
    
    def run_query(self, query, title=""):
        """Run SQL query and display results"""
        if title:
            print(f"\n{'='*60}")
            print(f"{title}")
            print(f"{'='*60}")
        
        try:
            result = pd.read_sql_query(query, self.conn)
            print(result.to_string(index=False))
            return result
        except Exception as e:
            print(f"Error in query: {e}")
            return None
    
    def export_to_csv(self, query, filename):
        """Export query results to CSV"""
        result = pd.read_sql_query(query, self.conn)
        result.to_csv(filename, index=False)
        print(f"✓ Exported to {filename}")
        return result
    
    def close(self):
        self.conn.close()

# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    print("HOUSE PRICE SQL ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SQLHouseAnalyzer()
    
    # Define your key SQL queries
    queries = {
        "market_overview": """
            SELECT 
                COUNT(*) AS total_properties,
                ROUND(AVG(price), 2) AS avg_price,
                ROUND(MIN(price), 2) AS min_price,
                ROUND(MAX(price), 2) AS max_price,
                ROUND(AVG(CASE WHEN price >= 650000 THEN price END), 2) AS high_value_avg_price
            FROM house_data;
        """,
        
        "high_value_analysis": """
            SELECT 
                CASE WHEN price >= 650000 THEN 'High Value (≥$650K)' ELSE 'Regular (<$650K)' END AS property_type,
                COUNT(*) AS count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS market_share_pct,
                ROUND(AVG(price), 2) AS avg_price,
                ROUND(AVG(sqft_living), 2) AS avg_living_area,
                ROUND(AVG(grade), 2) AS avg_grade,
                ROUND(SUM(CASE WHEN waterfront = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS waterfront_pct
            FROM house_data
            GROUP BY property_type;
        """,
        
        "price_category_distribution": """
            SELECT 
                CASE 
                    WHEN price < 300000 THEN '< $300K'
                    WHEN price BETWEEN 300000 AND 500000 THEN '$300K-500K'
                    WHEN price BETWEEN 500000 AND 650000 THEN '$500K-650K'
                    WHEN price BETWEEN 650000 AND 1000000 THEN '$650K-1M'
                    ELSE '> $1M'
                END AS price_category,
                COUNT(*) AS property_count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage,
                ROUND(AVG(price), 2) AS avg_price_in_category
            FROM house_data
            GROUP BY price_category
            ORDER BY avg_price_in_category;
        """,
        
        "top_correlations": """
            WITH correlations AS (
                SELECT 'sqft_living' AS feature, CORR(price, sqft_living) AS correlation FROM house_data
                UNION ALL SELECT 'grade', CORR(price, grade) FROM house_data
                UNION ALL SELECT 'bathrooms', CORR(price, bathrooms) FROM house_data
                UNION ALL SELECT 'sqft_above', CORR(price, sqft_above) FROM house_data
                UNION ALL SELECT 'view', CORR(price, view) FROM house_data
            )
            SELECT 
                feature,
                ROUND(correlation, 4) AS correlation_with_price,
                CASE 
                    WHEN ABS(correlation) >= 0.7 THEN 'Very Strong'
                    WHEN ABS(correlation) >= 0.5 THEN 'Strong'
                    WHEN ABS(correlation) >= 0.3 THEN 'Moderate'
                    WHEN ABS(correlation) >= 0.1 THEN 'Weak'
                    ELSE 'Very Weak'
                END AS strength
            FROM correlations
            ORDER BY ABS(correlation) DESC;
        """,
        
        "waterfront_premium": """
            SELECT 
                CASE WHEN waterfront = 1 THEN 'Waterfront' ELSE 'Non-Waterfront' END AS waterfront_status,
                COUNT(*) AS property_count,
                ROUND(AVG(price), 2) AS avg_price,
                ROUND((AVG(price) / (SELECT AVG(price) FROM house_data WHERE waterfront = 0) - 1) * 100, 2) AS premium_pct
            FROM house_data
            GROUP BY waterfront_status;
        """,
        
        "grade_impact": """
            SELECT 
                grade,
                COUNT(*) AS property_count,
                ROUND(AVG(price), 2) AS avg_price,
                ROUND(AVG(sqft_living), 2) AS avg_living_area,
                ROUND(SUM(CASE WHEN price >= 650000 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS high_value_pct
            FROM house_data
            GROUP BY grade
            ORDER BY grade;
        """
    }
    
    # Run all queries
    for name, query in queries.items():
        analyzer.run_query(query, name.replace('_', ' ').title())
    
    # Export key insights
    analyzer.export_to_csv(queries["high_value_analysis"], "data/high_value_analysis.csv")
    analyzer.export_to_csv(queries["price_category_distribution"], "data/price_categories.csv")
    
    # Close connection
    analyzer.close()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE! Files saved to /data/ directory")
    print("="*60)