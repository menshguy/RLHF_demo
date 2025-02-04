from flask import Flask, render_template
import sqlite3
import os

app = Flask(__name__)

def get_feedback_count():
    db_path = os.path.join(os.path.dirname(__file__), 'feedback.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM feedback")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_top_contributor():
    db_path = os.path.join(os.path.dirname(__file__), 'feedback.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT email, COUNT(*) as submission_count
        FROM feedback
        GROUP BY email
        ORDER BY submission_count DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'email': result[0],
            'count': result[1]
        }
    return None

@app.route('/')
def index():
    count = get_feedback_count()
    # Calculate percentage (max is 5000)
    percentage = min((count / 5000) * 100, 100)
    
    # Define milestones
    milestones = [
        {'value': 100, 'percentage': (100/5000)*100},
        {'value': 500, 'percentage': (500/5000)*100},
        {'value': 1000, 'percentage': (1000/5000)*100},
        {'value': 5000, 'percentage': 100}
    ]
    
    # Get top contributor
    top_contributor = get_top_contributor()
    
    return render_template('index.html', 
                         count=count, 
                         percentage=percentage, 
                         milestones=milestones,
                         top_contributor=top_contributor)

if __name__ == '__main__':
    app.run(debug=True)
