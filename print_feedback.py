import sqlite3
import pandas as pd
from tabulate import tabulate

def view_feedback():
    conn = sqlite3.connect('feedback.db')
    
    # Read the feedback data into a pandas DataFrame
    query = """
    SELECT 
        timestamp,
        substr(job_text, 1, 50) || '...' as job_preview,
        candidate_id,
        overall_score,
        location_match,
        skills_match,
        title_relevance,
        experience_match
    FROM feedback
    ORDER BY timestamp DESC;
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Rename columns for better readability
    df.columns = [
        'Timestamp',
        'Job Description (Preview)',
        'Candidate ID',
        'Overall Score',
        'Location Match',
        'Skills Match',
        'Title Relevance',
        'Experience Match'
    ]
    
    # Print the data in a nice table format
    print("\n=== Feedback Database Contents ===\n")
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    conn.close()

if __name__ == "__main__":
    view_feedback()
