from flask import Flask
from chatbot_interview import chatbot_bp
from resume_analyzer import resume_analyzer_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
app.register_blueprint(resume_analyzer_bp, url_prefix='/resume_analyzer')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
