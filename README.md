# AI Recruitment Assistant with RLHF

This is a Gradio-based web application that helps match job descriptions with candidate profiles using vector similarity search and allows for continuous improvement through human feedback.

## Features

- Vector-based semantic search using Sentence Transformers
- Synthetic data generation for testing (job descriptions and candidate profiles)
- Human feedback collection for model improvement
- Simple and intuitive web interface
- SQLite database for storing feedback data

## Installation

1. Clone this repository
2. Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. Enter a job description in the text box or use one of the example job descriptions provided

4. Click "Search Candidates" to find the top 3 matching candidates

5. To provide feedback:
   - Note the Candidate ID from the results
   - Enter the Candidate ID in the feedback section
   - Rate the match (0 for poor match, 1 for good match)
   - Click "Submit Feedback"

## How it Works

1. **Vector Embeddings**: The application uses the Sentence-BERT model 'all-MiniLM-L6-v2' to convert text descriptions into vector embeddings

2. **Similarity Search**: Cosine similarity is used to find the most similar candidates to a given job description

3. **Feedback Collection**: User feedback is stored in an SQLite database along with the job description embeddings for future model fine-tuning

4. **Synthetic Data**: The application uses the Faker library to generate realistic-looking job descriptions and candidate profiles for testing

## Future Improvements

1. Implement actual model fine-tuning using collected feedback data
2. Add more sophisticated candidate profile generation
3. Implement batch feedback collection
4. Add visualization of similarity scores
5. Add export functionality for feedback data
