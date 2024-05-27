import spacy
from pdfminer.high_level import extract_text
import re

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

def secure_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
