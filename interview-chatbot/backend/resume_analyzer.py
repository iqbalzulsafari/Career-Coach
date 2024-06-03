import spacy
from pdfminer.high_level import extract_text
import re
import os
from flask import Blueprint, request, jsonify
import joblib
import scipy.sparse

resume_analyzer_bp = Blueprint('resume_analyzer', __name__)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
classifier_path = r"interview-chatbot\\pkl\\nlp_model2.pkl"
vectorizer_skills_path = r"interview-chatbot\\pkl\\tfidf_vectorizer_skills.pkl"
vectorizer_tools_path = r"interview-chatbot\\pkl\\tfidf_vectorizer_tools.pkl"
vectorizer_education_path = r"interview-chatbot\\pkl\\tfidf_vectorizer_education.pkl"
job_positions_path = r"interview-chatbot\\pkl\\job_positions.pkl"

classifier = joblib.load(classifier_path)
vectorizer_skills = joblib.load(vectorizer_skills_path)
vectorizer_tools = joblib.load(vectorizer_tools_path)
vectorizer_education = joblib.load(vectorizer_education_path)
job_positions = joblib.load(job_positions_path)

# Define Flask app
app = Flask(__name__)

def secure_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

# Define function to extract text from PDF
def extract_text_from_pdf(pdf):
    return extract_text(pdf)

# Define function to clean text
def clean_text(text):
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra whitespace
    return cleaned_text

#----------------------------------------------------------------------------------------

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(filename)
    
    text = extract_text(filename)
    os.remove(filename)
    
    # Extract skills and experiences
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    experiences = [ent.text for ent in doc.ents if ent.label_ == "EXPERIENCE"]
    
    return jsonify({"skills": skills, "experiences": experiences})

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

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    # Receive resume file and desired job from the user
    file = request.files['file']
    desired_job = request.form['desired_job']
    filename = secure_filename(file.filename)
    file.save(filename)
    
    # Extract text from the resume PDF
    text = extract_text(filename)
    os.remove(filename)
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Analyze the resume components and evaluate
    components = ["Personal Info", "Objective / Summary", "Education", "Work / Experience", 
                  "Awards and Honours", "Skills", "Language", "Reference"]
    scores = {}
    for component in components:
        score = evaluate_component(cleaned_text, component)
        scores[component] = score
    
    # Calculate total score
    total_score = sum(scores.values())
    
    # Analyze the quality of the resume for the desired job
    resume_quality = analyze_quality(total_score)

    # Analyze the resume using the trained NLP model
    resume_category = analyze_resume_category(cleaned_text)
    
    return jsonify({"scores": scores, "total_score": total_score, "resume_quality": resume_quality})

#----------------------------------------------------------------------------------------

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

# Streamlit app
def main():
    st.title("Resume Analyzer")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        selected_job = st.selectbox("Select the job position you want to apply for", job_positions)
        if st.button("Analyze"):
            scores, total_score, resume_quality, resume_category = analyze_resume(resume_text)
            fit_score = analyze_job_fit(resume_text, selected_job)
            st.write("Your desired job is: ", selected_job)
            st.write("Scores for each component:")
            st.write(scores)
            st.write("Total Score:", total_score)
            st.write("Resume Quality:", resume_quality)
            st.write(fit_score)
            st.write("Based on your resume, your ideal job is:", resume_category)

if __name__ == "__main__":
    app.run(debug=True)