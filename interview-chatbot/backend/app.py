from flask import Flask
from chatbot_interview import chatbot_bp
from resume_analyzer import resume_analyzer_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
app.register_blueprint(resume_analyzer_bp, url_prefix='/resume_analyzer')

if __name__ == "__main__":
    app.run(debug=True)

