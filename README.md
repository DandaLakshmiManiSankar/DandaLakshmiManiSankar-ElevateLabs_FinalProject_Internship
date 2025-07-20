# DandaLakshmiManiSankar-ElevateLabs_FinalProject_Internship

This project ranks resumes based on their relevance to a job description using NLP techniques like TF-IDF and SpaCy. Built using Python, Flask, and Scikit-learn, it helps HR teams screen resumes efficiently by scoring and ranking candidates.

## Features

1.Extract text from PDF resumes

2.Clean and preprocess with SpaCy

3.Match resumes to job descriptions using TF-IDF similarity

4.Score and rank candidates

5.Web-based UI to upload resumes and view results

6.Option to download HR report (CSV)

# Installation & Setup

## Clone the Repository

git clone https://github.com/DandaLakshmiManiSankar/DandaLakshmiManiSankar/ElevateLabs_FinalProject_Internship.git
cd ai-resume-ranker

## Create a Virtual Environment

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

## Install Dependencies

pip install -r requirements.txt

python -m spacy download en_core_web_sm

## Running the App

python app.py


