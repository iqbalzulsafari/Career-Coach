import os
import openai
import spacy
from transformers import BertModel, BertTokenizer, pipeline
import torch
import pandas as pd

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

def evaluate_answer(answer, ideal_answer):
    # Compute semantic similarity
    answer_embedding = semantic_similarity(answer)
    ideal_answer_embedding = semantic_similarity(ideal_answer)
    
    similarity_score = torch.cosine_similarity(
        torch.tensor(answer_embedding).mean(dim=1),
        torch.tensor(ideal_answer_embedding).mean(dim=1)
    ).item()

    # Provide feedback based on the similarity score
    if similarity_score > 0.8:
        feedback = "High"
    elif similarity_score > 0.6:
        feedback = "Medium"
    else:
        feedback = "Low"

    return similarity_score, feedback

def main():
    # Load the interview dataset
    dataset_path = "C:/Users/iqbalzulsafari/Documents/NLP-Chatbot/interview-chatbot/Dataset/Interview/Interview_Questions.csv"
    dataset = pd.read_csv(dataset_path)

    # Get unique categories from the dataset
    categories = dataset["category"].unique()

    # Ask for the user's course (category)
    print("Available categories:")
    for i, category in enumerate(categories, 1):
        print(f"{i}. {category}")
    category_choice = int(input("Choose a category (enter the number): "))
    selected_category = categories[category_choice - 1]

    # Filter dataset based on selected category
    category_dataset = dataset[dataset["category"] == selected_category]

    # Start the interview for the category
    print(f"Welcome to the interview for the {selected_category} course!")
    print("Type 'exit' at any time to end the interview.")

    # Iterate over questions for the category
    for index, row in category_dataset.iterrows():
        question = row["questions"]
        best_answer = row["best answer"]

        print("\nQuestion:", question)
        answer = input("Your answer: ")
        if answer.lower() == 'exit':
            print("Interview ended.")
            return

        score, feedback = evaluate_answer(answer, best_answer)
        print("Similarity score:", score)
        print("Evaluation:", feedback)

if __name__ == "__main__":
    main()
