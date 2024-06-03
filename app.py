from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import chatbot_interview
import resume_analyzer

app = Flask(__name__)

# Route for the main page
@app.route('/')
def main_page():
    return render_template('main_page.html')

# Route for the mockup interview page
@app.route('/mockup_interview')
def mockup_interview():
    return render_template('ChatBot.html')

# Route for handling chatbot interaction
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    message = data['message']
    # Process the message using the chatbot
    response = chatbot_interview.process_message(message)
    return jsonify({'response': response})

# Route for the resume analyzer page
@app.route('/resume_analyzer')
def resume_analyzer():
    return render_template('resume1.html')

# Route for uploading resume
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({"filename": filename})

# Route for analyzing resume and providing feedback
@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    filename = request.form['filename']
    # Process the resume using the resume analyzer
    feedback = resume_analyzer.process_resume(filename)
    return jsonify({"feedback": feedback})

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'  # Specify the upload folder
    app.run(debug=True)
