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
            ))
        }
        candidates.append(candidate)
    return candidates

def generate_job_descriptions(n_jobs=20):
    jobs = []
    for _ in range(n_jobs):
        job = {
            'title': fake.job(),
            'description': fake.text(max_nb_chars=300),
            'requirements': fake.text(max_nb_chars=200)
        }
        jobs.append(job)
    return jobs

# Initialize database
def init_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (timestamp TEXT, job_text TEXT, candidate_id INTEGER, 
                  score INTEGER, embedding BLOB)''')
    conn.commit()
    conn.close()

# Vector search functionality
class VectorSearch:
    def __init__(self):
        self.candidates = generate_candidate_data()
        self.candidate_texts = [
            f"{c['skills']} {c['experience']} {c['education']}" 
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
def save_feedback(job_text, candidate_id, score):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    embedding = model.encode([job_text])[0].tobytes()
    c.execute('''INSERT INTO feedback VALUES (?, ?, ?, ?, ?)''',
              (datetime.now().isoformat(), job_text, candidate_id, score, embedding))
    conn.commit()
    conn.close()

# Gradio interface
def search_candidates(job_description):
    results = vector_search.search(job_description)
    output = []
    for candidate, score in results:
        output.append(f"Match Score: {score:.2f}\n")
        output.append(f"Name: {candidate['name']}\n")
        output.append(f"Skills: {candidate['skills']}\n")
        output.append(f"Experience: {candidate['experience']}\n")
        output.append(f"Education: {candidate['education']}\n")
        output.append("-" * 50 + "\n")
    return "".join(output)

def submit_feedback(job_desc, candidate_id, score):
    save_feedback(job_desc, candidate_id, score)
    return f"Feedback saved for candidate {candidate_id}"

# Example job descriptions
example_jobs = generate_job_descriptions(5)

# Create Gradio interface
demo = gr.Interface(
    fn=search_candidates,
    inputs=[
        gr.Textbox(
            label="Job Description",
            placeholder="Enter job description here...",
            lines=5
        )
    ],
    outputs=gr.Textbox(label="Matching Candidates", lines=10),
    examples=[[job['description']] for job in example_jobs],
    title="AI Recruitment Assistant",
    description="Enter a job description to find matching candidates based on their skills and experience."
)

# Add feedback interface
demo2 = gr.Interface(
    fn=submit_feedback,
    inputs=[
        gr.Textbox(label="Job Description"),
        gr.Number(label="Candidate ID for Feedback"),
        gr.Slider(minimum=0, maximum=1, step=1, label="Rate Match (0: Poor, 1: Good)")
    ],
    outputs=gr.Textbox(label="Feedback Status"),
    title="Submit Feedback",
    description="Submit feedback for a candidate."
)

# Initialize database
init_db()

# Launch the app
if __name__ == "__main__":
    demo.launch()
    demo2.launch()
