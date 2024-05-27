import spacy
from pdfminer.high_level import extract_text
from docx import Document

nlp = spacy.load('en_core_web_sm')

def extract_text_from_file(file):
    if file.filename.endswith('.pdf'):
        text = extract_text(file)
    elif file.filename.endswith('.docx'):
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format.")
    return text

def analyze_resume(file):
    text = extract_text_from_file(file)
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
    experiences = [ent.text for ent in doc.ents if ent.label_ == 'EXPERIENCE']
    education = [ent.text for ent in doc.ents if ent.label_ == 'EDUCATION']

    recommendations = {
        "skills": skills,
        "experiences": experiences,
        "education": education,
        "overall_feedback": "Ensure your resume is concise and highlights relevant skills and experiences."
    }
    return recommendations
