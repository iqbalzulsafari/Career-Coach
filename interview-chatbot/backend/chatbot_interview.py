import os
import openai
import spacy
from transformers import BertModel, BertTokenizer, pipeline
import torch
from datasets import load_dataset

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
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=50,
    )
    question = response.choices[0].text.strip()
    return question

def evaluate_answer(answer, responses, evaluations):
    max_similarity_score = -1
    closest_response = None
    feedback = None
    
    # Compute semantic similarity with each response in the dataset
    for response, evaluation in zip(responses, evaluations):
        similarity_score = torch.cosine_similarity(
            torch.tensor(semantic_similarity(answer)).mean(dim=1),
            torch.tensor(semantic_similarity(response)).mean(dim=1)
        ).item()
        
        # Keep track of the closest match
        if similarity_score > max_similarity_score:
            max_similarity_score = similarity_score
            closest_response = response
            feedback = evaluation
    
    return max_similarity_score, closest_response, feedback

def main():
    # Load the interview dataset
    dataset = load_dataset("waelChafei/interviewtest")

    # Extract questions, responses, and evaluations
    questions = dataset["train"]["question"]
    responses = dataset["train"]["response"]
    evaluations = dataset["train"]["evaluation"]

    # Ask for the user's role
    role = input("Enter your role: ")

    # Start the interview
    print("Welcome to the interview, " + role + "!")
    print("Type 'exit' at any time to end the interview.")

    for question, response, evaluation in zip(questions, responses, evaluations):
        print("\nQuestion:", question)
        answer = input("Your answer: ")
        if answer.lower() == 'exit':
            print("Interview ended.")
            break

        score, closest_response, feedback = evaluate_answer(answer, responses, evaluations)
        print("Similarity score:", score)
        print("Ideal answer:", closest_response)
        print("Answer evaluation:", feedback)
    else:
        print("Interview completed. Thank you!")

if __name__ == "__main__":
    main()
