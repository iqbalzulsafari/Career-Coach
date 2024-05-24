from flask import Flask, request, jsonify
import openai
import spacy
from transformers import BertModel, BertTokenizer, pipeline
import torch

app = Flask(__name__)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
semantic_similarity = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

# Set OpenAI API key
openai.api_key = 'sk-proj-QNMLFaZleFiasA9oTcXPT3BlbkFJSMiaO2D9WBaFiwEHgR6p'

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    role = data.get('role')
    
    # Generate question using OpenAI GPT
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=f"Generate an interview question for a {role} position.",
        max_tokens=50
    )
    question = response.choices[0].text.strip()
    return jsonify({"question": question})

@app.route('/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.json
    answer = data.get('answer')
    ideal_answer = data.get('ideal_answer')

    # Compute semantic similarity
    answer_embedding = semantic_similarity(answer)
    ideal_answer_embedding = semantic_similarity(ideal_answer)
    
    similarity_score = torch.cosine_similarity(
        torch.tensor(answer_embedding).mean(dim=1),
        torch.tensor(ideal_answer_embedding).mean(dim=1)
    ).item()

    feedback = "Good job!" if similarity_score > 0.8 else "Needs improvement."
    return jsonify({"score": similarity_score, "feedback": feedback})

if __name__ == "__main__":
    app.run(debug=True)

