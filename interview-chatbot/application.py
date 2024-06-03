from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main_page.html')

@app.route('/resume_analyzer')
def resume_analyzer():
    return render_template('resume1.html')

@app.route('/resume_analyzer_result')
def resume_analyzer_result():
    return render_template('resume2.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Handle chatbot requests here
    # Example: process input, generate response, and return it
    data = request.json
    user_message = data['message']
    # Process user_message and generate bot_response
    bot_response = "This is a sample response from the chatbot."
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
