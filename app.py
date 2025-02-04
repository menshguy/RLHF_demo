import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from faker import Faker
import json
import sqlite3
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Initialize the model and faker
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
fake = Faker()

# Generate synthetic data
def generate_candidate_data(n_candidates=100):
    candidates = []
    for _ in range(n_candidates):
        candidate = {
            'id': _,
            'name': fake.name(),
            'location': fake.city(),
            'skills': ', '.join(fake.random_elements(elements=(
                'Python', 'JavaScript', 'Machine Learning', 'Data Science',
                'Web Development', 'DevOps', 'Cloud Computing', 'SQL',
                'Project Management', 'Agile', 'Communication', 'Leadership'
            ), length=random.randint(3, 6))),
            'experience': fake.text(max_nb_chars=200),
            'education': fake.random_element(elements=(
                'Bachelor in Computer Science',
                'Master in Data Science',
                'PhD in Machine Learning',
                'Bachelor in Engineering',
                'Master in Business Administration'
            )),
            'years_experience': random.randint(1, 15)
        }
        candidates.append(candidate)
    return candidates

def generate_job_descriptions(n_jobs=20):
    jobs = []
    for _ in range(n_jobs):
        job = {
            'title': fake.job(),
            'location': fake.city(),
            'description': fake.text(max_nb_chars=300),
            'requirements': fake.text(max_nb_chars=200),
            'years_required': random.randint(1, 10)
        }
        jobs.append(job)
    return jobs

# Initialize database
def init_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    
    # Drop the table if it exists
    c.execute('DROP TABLE IF EXISTS feedback')
    
    # Create the table with all our feedback columns
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (timestamp TEXT,
                  job_text TEXT,
                  candidate_id INTEGER,
                  overall_score INTEGER,
                  location_match INTEGER,
                  skills_match INTEGER,
                  title_relevance INTEGER,
                  experience_match INTEGER,
                  embedding BLOB)''')
    conn.commit()
    conn.close()

# Vector search functionality
class VectorSearch:
    def __init__(self):
        self.candidates = generate_candidate_data()
        self.candidate_texts = [
            f"{c['skills']} {c['experience']} {c['education']} {c['location']} {c['years_experience']} years" 
            for c in self.candidates
        ]
        self.embeddings = model.encode(self.candidate_texts)

    def search(self, query, k=3):
        query_vector = model.encode([query])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(self.candidates[idx], float(similarities[idx])) 
                for idx in top_k_indices]

# Initialize vector search
vector_search = VectorSearch()

# Save feedback
def save_feedback(job_text, candidate_id, overall_score, location_match, 
                 skills_match, title_relevance, experience_match):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    embedding = model.encode([job_text])[0].tobytes()
    c.execute('''INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().isoformat(), job_text, candidate_id, 
               overall_score, location_match, skills_match, 
               title_relevance, experience_match, embedding))
    conn.commit()
    conn.close()

# Gradio interface
def format_candidate_choice(candidate, score):
    return f"ID {candidate['id']} - {candidate['name']} (Match Score: {score:.2f})"

def search_candidates(job_description):
    results = vector_search.search(job_description)
    output_text = []
    choices = []
    
    for candidate, score in results:
        output_text.append(f"Match Score: {score:.2f}\n")
        output_text.append(f"Candidate ID: {candidate['id']}\n")
        output_text.append(f"Name: {candidate['name']}\n")
        output_text.append(f"Location: {candidate['location']}\n")
        output_text.append(f"Skills: {candidate['skills']}\n")
        output_text.append(f"Experience: {candidate['experience']}\n")
        output_text.append(f"Years of Experience: {candidate['years_experience']}\n")
        output_text.append(f"Education: {candidate['education']}\n")
        output_text.append("-" * 50 + "\n")
        
        choices.append(format_candidate_choice(candidate, score))
    
    return "".join(output_text), gr.Dropdown(choices=choices, label="Select Candidate for Feedback", interactive=True)

def get_candidate_id(candidate_choice):
    if not candidate_choice:
        return None
    # Extract ID from the format "ID X - Name (Match Score: Y.YY)"
    return int(candidate_choice.split()[1])

def submit_feedback(job_desc, candidate_id, overall_score, 
                   location_match, skills_match, title_relevance, experience_match):
    try:
        save_feedback(job_desc, int(candidate_id), int(overall_score),
                     int(location_match), int(skills_match),
                     int(title_relevance), int(experience_match))
        return "Feedback saved successfully!"
    except Exception as e:
        return f"Error saving feedback: {str(e)}"

# Example job descriptions
example_jobs = generate_job_descriptions(5)

# Create Gradio interface
with gr.Blocks(title="AI Recruitment Assistant") as demo:
    gr.Markdown("# AI Recruitment Assistant")
    
    with gr.Row():
        with gr.Column():
            job_input = gr.Textbox(
                label="Job Description",
                placeholder="Enter job description here...",
                lines=5
            )
            search_btn = gr.Button("Search Candidates")
            
            # Example selector
            gr.Examples(
                examples=[[job['description']] for job in example_jobs],
                inputs=job_input,
                label="Example Job Descriptions"
            )
    
    results_output = gr.Textbox(
        label="Matching Candidates",
        lines=10,
        interactive=False
    )
    
    candidate_dropdown = gr.Dropdown(
        label="Select Candidate for Feedback",
        interactive=True
    )
    
    gr.Markdown("## Provide Feedback")
    with gr.Row():
        with gr.Column():
            candidate_id = gr.Number(label="Selected Candidate ID", interactive=False)
            overall_score = gr.Slider(minimum=1, maximum=5, step=1, label="Overall Match Score (1-5)")
        with gr.Column():
            location_match = gr.Radio(choices=[1, 2, 3, 4, 5], label="Location Match (1: Poor - 5: Excellent)")
            skills_match = gr.Radio(choices=[1, 2, 3, 4, 5], label="Skills Match (1: Poor - 5: Excellent)")
            title_relevance = gr.Radio(choices=[1, 2, 3, 4, 5], label="Job Title Relevance (1: Poor - 5: Excellent)")
            experience_match = gr.Radio(choices=[1, 2, 3, 4, 5], label="Experience Level Match (1: Poor - 5: Excellent)")
    
    feedback_btn = gr.Button("Submit Feedback")
    feedback_output = gr.Textbox(label="Feedback Status")
    
    # Connect components
    search_outputs = [results_output, candidate_dropdown]
    search_btn.click(
        fn=search_candidates,
        inputs=job_input,
        outputs=search_outputs
    )
    
    # Update candidate ID when dropdown selection changes
    candidate_dropdown.change(
        fn=get_candidate_id,
        inputs=[candidate_dropdown],
        outputs=[candidate_id]
    )
    
    feedback_btn.click(
        fn=submit_feedback,
        inputs=[
            job_input,
            candidate_id,
            overall_score,
            location_match,
            skills_match,
            title_relevance,
            experience_match
        ],
        outputs=feedback_output
    )

# Initialize database
init_db()

# Launch the app
if __name__ == "__main__":
    demo.launch()
