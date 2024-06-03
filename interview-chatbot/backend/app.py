import os
import openai
import pandas as pd
import spacy
import joblib
from flask import Flask, request, jsonify
from transformers import BertModel, BertTokenizer, pipeline
import torch
import scipy.sparse
from pdfminer.high_level import extract_text
import streamlit as st
import re

# Load NLP models
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
semantic_similarity = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

# Set OpenAI API key
openai.api_key = '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'

# Initialize Flask app
app = Flask(__name__)

# Define function to clean text
def clean_text(text):
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra whitespace
    return cleaned_text

# Define function to extract text from PDF
def extract_text_from_pdf(pdf):
    return extract_text(pdf)

# Define function to evaluate answer
def evaluate_answer(answer, ideal_answer):
    # Compute semantic similarity
    answer_embedding = semantic_similarity(answer)
    ideal_answer_embedding = semantic_similarity(ideal_answer)
    
    similarity_score = torch.cosine_similarity(
        torch.tensor(answer_embedding).mean(dim=1),
        torch.tensor(ideal_answer_embedding).mean(dim=1)
    ).item() * 10  # Scale similarity score to /10 form

    # Provide feedback based on the similarity score
    if similarity_score > 8.5:
        feedback = "High"
    elif similarity_score > 5:
        feedback = "Medium"
    else:
        feedback = "Low"

    return similarity_score, feedback

# Define function to evaluate resume component
def evaluate_component(text, component):
    score = 0
    keywords = {
        "Personal Info": ["name", "address", "contact", "email"],
        "Education": ["education", "degree", "qualification"],
        "Work / Experience": ["work", "experience", "job"],
        "Awards and Honours": ["award", "honour"],
        "Skills": ["skill"],
        "Language": ["language"],
        "Reference": ["reference"]
    }
    detected_components = 0  # To count the number of detected components
    
    for keyword in keywords[component]:
        if keyword in text:
            detected_components += 1
            score = 10
    
    return score

# Define function to analyze resume quality
def analyze_quality(scores, total_score):
    missing_components = [component for component, score in scores.items() if score == 0]
    if not missing_components and total_score == 100:
        return "Excellent! Your resume has complete details."
    elif not missing_components:
        return "Good! Your resume contains all necessary components."
    else:
        if total_score >= 50:
            return f"Fair. Consider adding the following components to your resume: {', '.join(missing_components)}"
        else:
            return f"Poor. Missing critical components: {', '.join(missing_components)}"

# Define function to analyze resume
def analyze_resume(text):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize the text using the pre-trained TF-IDF vectorizers
    vec_skills = vectorizer_skills.transform([cleaned_text])
    vec_tools = vectorizer_tools.transform([cleaned_text])
    vec_education = vectorizer_education.transform([cleaned_text])
    
    # Concatenate feature vectors
    vec = scipy.sparse.hstack([vec_skills, vec_tools, vec_education])
    
    # Predict the job category
    category = classifier.predict(vec)[0]

    # Analyze the resume components and evaluate
    components = ["Personal Info", "Education", "Work / Experience", 
                  "Awards and Honours", "Skills", "Language", "Reference"]
    scores = {}
    total_score = 0  # Initialize total score
    for component in components:
        score = evaluate_component(cleaned_text, component)
        scores[component] = score
        total_score += score  # Accumulate scores
    
    # Calculate total score as a percentage
    total_score_percentage = (total_score / (len(components) * 10)) * 100
    
    # Analyze the quality of the resume for the desired job
    resume_quality = analyze_quality(scores, total_score_percentage)
    
    return scores, total_score_percentage, resume_quality, category

# Define function to analyze job fit score
def analyze_job_fit(text, job_position):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize the text using the pre-trained TF-IDF vectorizers
    vec_skills = vectorizer_skills.transform([cleaned_text])
    vec_tools = vectorizer_tools.transform([cleaned_text])
    vec_education = vectorizer_education.transform([cleaned_text])
    
    # Concatenate feature vectors
    vec = scipy.sparse.hstack([vec_skills, vec_tools, vec_education])
    
    # Predict the job fit using the trained classifier
    predicted_job = classifier.predict(vec)[0]
    fit_score = 1 if predicted_job == job_position else 0  # Simplified fit score
    fit_score = ""
    if predicted_job == job_position:
        fit_score = "Congratulations! your resume fit your desired job."
    else:
        fit_score = "Unfortunately, your resume does not fit for {}".format(job_position)
    
    return fit_score

# Define route to conduct chatbot interview
@app.route('/conduct_chatbot_interview', methods=['GET'])
def conduct_chatbot_interview():
    # Load the interview dataset
    dataset_path = "C:/Users/iqbalzulsafari/Documents/NLP-Chatbot/interview-chatbot/Dataset/Interview_Questions.csv"
    dataset = pd.read_csv(dataset_path)

    # Get unique categories from the dataset
    categories = dataset["category"].unique()

    # Ask for the user's course (category)
    print("Available categories:")
    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")
    category_choice = int(input("Choose a category (enter the number): "))
    selected_category = categories[category_choice - 1]

    # Filter dataset based on selected category
    category_dataset = dataset[dataset["category"] == selected_category]

    # Start the interview for the category
    print(f"Welcome to the interview for the {selected_category} course!")
    print("Type 'exit' at any time to end the interview.")

    total_score = 0
    num_questions = len(category_dataset)
    low_or_medium_count = 0
    
    # Iterate over questions for the category
    for index, row in category_dataset.iterrows():
        question = row["questions"]
        best_answer = row["best answer"]

        print("\nQuestion:", question)
        answer = input("Your answer: ")
        if answer.lower() == 'exit':
            print("Interview ended.")
            return

        score, feedback = evaluate_answer(answer, best_answer)
        print("Score (/10): {:.1f}".format(score))
        print("Answer evaluation:", feedback)
        
        total_score += score
        
        if feedback in ["Low", "Medium"]:
            low_or_medium_count += 1
            print("Best answer:", best_answer)

    overall_score = total_score / num_questions
    overall_score_scaled = overall_score
    print("\nOverall Score (/10): {:.1f}".format(overall_score_scaled))
    print("Interview completed. Thank you!")
    
# Define route to analyze resume
@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    data = request.json
    text = data.get('text')
    
    # Analyze the resume using the trained NLP model
    scores, total_score, resume_quality, resume_category = analyze_resume(text)
    
    return jsonify({"scores": scores, "total_score": total_score, "resume_quality": resume_quality, "resume_category": resume_category})

# Define route to analyze resume feedback
@app.route('/resume_feedback', methods=['POST'])
def resume_feedback():
    data = request.json
    text = data.get('text')
    
    # Analyze text and provide feedback
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    feedback = []
    if not any(ent[1] == "SKILL" for ent in entities):
        feedback.append("Consider adding relevant skills.")
    if not any(ent[1] == "EXPERIENCE" for ent in entities):
        feedback.append("Consider detailing your experiences.")
    
    return jsonify({"feedback": feedback})

# Define route to upload resume
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)
    
    text = extract_text_from_pdf(filename)
    
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(debug=True)
