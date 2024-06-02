from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')

    # Your chatbot logic goes here
    # Example: Dummy response for now
    bot_response = "This is a dummy response from the chatbot."

    return jsonify({'response': bot_response})

@app.route('/resume-analyzer', methods=['POST'])
def resume_analyzer():
    # Your resume analyzer logic goes here
    # Example: Dummy response for now
    resume_data = request.json.get('resume')

    # Analyze the resume and generate a score
    resume_score = 90
    issues_found = 5

    return jsonify({'score': resume_score, 'issues_found': issues_found})

if __name__ == '__main__':
    app.run(debug=True)
