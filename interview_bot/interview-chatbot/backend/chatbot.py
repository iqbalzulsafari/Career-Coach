import os
import openai
import spacy
from transformers import BertModel, BertTokenizer, pipeline
import torch

# Load NLP models
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
semantic_similarity = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

# Set OpenAI API key
openai.api_key = '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'

def generate_question(role):
    # Generate question using OpenAI GPT
    prompt = f"Generate an interview question for a {role} position."
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=50,
    )
    question = response.choices[0].text.strip()
    return question

def evaluate_answer(answer):
    # Compute semantic similarity
    answer_embedding = semantic_similarity(answer)
    #ideal_answer_embedding = semantic_similarity(ideal_answer)
    
    similarity_score = torch.cosine_similarity(
        torch.tensor(answer_embedding).mean(dim=1),
        torch.tensor(answer_embedding).mean(dim=1) # Compare with itself (user's answer)
        #torch.tensor(ideal_answer_embedding).mean(dim=1)
    ).item()

    # Provide feedback based on the similarity score
    if similarity_score > 0.8:
        feedback = "Your answer closely matches the ideal answer. Well done!"
    elif similarity_score > 0.6:
        feedback = "Your answer is good, but there is room for improvement."
    else:
        feedback = "Your answer needs improvement. Try to provide more relevant information."

    return similarity_score, feedback
