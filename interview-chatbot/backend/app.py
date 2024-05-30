from flask import Flask, request, jsonify
from chatbot import generate_question, evaluate_answer

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    role = data.get('role')
    
    question = generate_question(role)
    return jsonify({"question": question})

@app.route('/evaluate', methods=['POST'])
def evaluate_answer_route():
    data = request.json
    answer = data.get('answer')
    ideal_answer = data.get('ideal_answer')

    score, feedback = evaluate_answer(answer, ideal_answer)
    return jsonify({"score": score, "feedback": feedback})

if __name__ == "__main__":
    app.run(debug=True)
