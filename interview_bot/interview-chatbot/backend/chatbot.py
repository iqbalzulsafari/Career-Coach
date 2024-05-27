import openai
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

openai.api_key = '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'

def generate_interview_question(role):
    prompt = f"Generate a common interview question for the role of {role}."
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=50)
    return response.choices[0].text.strip()

def evaluate_answer(question, answer):
    expected_answer = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Generate an ideal answer for the interview question: {question}",
        max_tokens=150
    ).choices[0].text.strip()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    question_embedding = model(**tokenizer(question, return_tensors='pt'))[0].mean(1)
    answer_embedding = model(**tokenizer(answer, return_tensors='pt'))[0].mean(1)
    expected_embedding = model(**tokenizer(expected_answer, return_tensors='pt'))[0].mean(1)

    similarity = 1 - cosine(answer_embedding.detach().numpy(), expected_embedding.detach().numpy())
    feedback = f"Similarity score: {similarity:.2f}. "

    if similarity > 0.8:
        feedback += "Great answer!"
    elif similarity > 0.5:
        feedback += "Good answer, but there's room for improvement."
    else:
        feedback += "Consider improving your answer."

    return feedback
