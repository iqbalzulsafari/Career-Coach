import streamlit as st
import requests

backend_url = "http://backend:5000"

st.title("Job Application Assistant")

st.sidebar.title("Navigation")
options = ["Interview Assistant", "Resume Analyzer"]
choice = st.sidebar.selectbox("Select an option:", options)

if choice == "Interview Assistant":
    st.header("Interview Assistant")
    
    role = st.text_input("Enter the job role:")
    if st.button("Start Interview"):
        response = requests.post(f"{backend_url}/chatbot/start_interview", json={"role": role})
        question = response.json().get("question")
        st.write(f"Question: {question}")
        
        answer = st.text_input("Your Answer:")
        ideal_answer = st.text_input("Ideal Answer:")
        if st.button("Evaluate Answer"):
            evaluation_response = requests.post(f"{backend_url}/chatbot/evaluate", json={"answer": answer, "ideal_answer": ideal_answer})
            score = evaluation_response.json().get("score")
            feedback = evaluation_response.json().get("feedback")
            st.write(f"Score: {score}")
            st.write(f"Feedback: {feedback}")

elif choice == "Resume Analyzer":
    st.header("Resume Analyzer")
    
    resume_file = st.file_uploader("Upload your resume:", type=["pdf"])
    desired_job = st.text_input("Enter desired job role:")
    if resume_file and st.button("Analyze Resume"):
        files = {"file": resume_file}
        data = {"desired_job": desired_job}
        response = requests.post(f"{backend_url}/resume_analyzer/analyze_resume", files=files, data=data)
        result = response.json()
        st.write(f"Scores: {result.get('scores')}")
        st.write(f"Total Score: {result.get('total_score')}")
        st.write(f"Resume Quality: {result.get('resume_quality')}")
        st.write(f"Resume Category: {result.get('resume_category')}")
