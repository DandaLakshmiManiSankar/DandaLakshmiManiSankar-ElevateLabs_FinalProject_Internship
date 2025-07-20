import os
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv

# Output CSV file name
csv_filename = "ranked_resumes.csv"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample job description Template
job_description = "NLP Specialist: Develop and implement NLP algorithms. Proficiency in Python, NLP libraries, and AI/ML frameworks required. Skilled in Python. Should be known about Full Stack."

# Automatically detect all PDFs in current folder
resume_paths = [f for f in os.listdir() if f.endswith(".pdf")]

# Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Extract name and email
def extract_entities(text):
    emails = re.findall(r'\S+@\S+', text)
    names = re.findall(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', text)
    if names:
        names = [" ".join(names[0])]
    return emails, names

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

# Ranking logic
ranked_resumes = []
for path in resume_paths:
    resume_text = extract_text_from_pdf(path)
    if not resume_text.strip():
        continue  # Skip empty resumes

    emails, names = extract_entities(resume_text)
    resume_vector = tfidf_vectorizer.transform([resume_text])
    similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
    ranked_resumes.append((names, emails, similarity))

# Sort results
ranked_resumes.sort(key=lambda x: x[2], reverse=True)

# Print results
for rank, (names, emails, similarity) in enumerate(ranked_resumes, start=1):
    print(f"Rank {rank}: Name={names}, Email={emails}, Score={similarity:.2f}%")

# Save to CSV
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Rank", "Name", "Email", "Similarity"])
    for rank, (names, emails, similarity) in enumerate(ranked_resumes, start=1):
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        writer.writerow([rank, name, email, round(similarity, 2)])
