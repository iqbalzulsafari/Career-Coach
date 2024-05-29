from chatbot import generate_question, evaluate_answer

def main():
        # Ask for the user's role
        role = input("Enter your role: ")

        # Predefined set of interview questions
        interview_questions = [
            "Tell me about yourself.",
            "What are your strengths?",
            "What are your weaknesses?",
            "Why do you want to work here?",
            "Tell me about a challenge you faced at work and how you overcame it."
        ]

        # Start the interview
        print("Welcome to the interview, " + role + "!")
        print("Type 'exit' at any time to end the interview.")
    
        for question in interview_questions:
            print("\nQuestion:", question)
            answer = input("Your answer: ")
            if answer.lower() == 'exit':
                print("Interview ended.")
                break

            #ideal_answer = "Sample ideal answer for " + question  # You can provide ideal answers for each question if available
            score, feedback = evaluate_answer(answer)
            print("Similarity score:", score)
            print("Feedback:", feedback)
        else:
            print("Interview completed. Thank you!")

if __name__ == "__main__":
    main()
