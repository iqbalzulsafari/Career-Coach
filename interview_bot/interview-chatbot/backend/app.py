from flask import Flask, request, jsonify
from flask_cors import CORS
from models import db, setup_db
from interview_bot import generate_interview_question, evaluate_answer
from resume_analyzer import analyze_resume

app = Flask(__name__)
CORS(app)
setup_db(app)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    question = generate_interview_question(data['role'])
    return jsonify({"question": question})

@app.route('/evaluate_answer', methods=['POST'])
def evaluate():
    data = request.json
    feedback = evaluate_answer(data['question'], data['answer'])
    return jsonify({"feedback": feedback})

@app.route('/analyze_resume', methods=['POST'])
def analyze():
    file = request.files['resume']
    recommendations = analyze_resume(file)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
