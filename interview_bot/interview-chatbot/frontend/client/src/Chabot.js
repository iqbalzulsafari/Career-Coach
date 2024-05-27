import React, { useState } from 'react';
import axios from 'axios';

const Chatbot = () => {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [response, setResponse] = useState(null);

    const askQuestion = async () => {
        const res = await axios.post('/ask', { question, role: 'Software Engineer' });
        setQuestion(res.data.question);
    };

    const evaluateAnswer = async () => {
        const res = await axios.post('/evaluate', { answer, ideal_answer: 'Ideal Answer for the question' });
        setResponse(res.data);
    };

    return (
        <div>
            <button onClick={askQuestion}>Ask Question</button>
            <p>Question: {question}</p>
            <textarea value={answer} onChange={(e) => setAnswer(e.target.value)} />
            <button onClick={evaluateAnswer}>Submit Answer</button>
            {response && (
                <div>
                    <p>Score: {response.score}</p>
                    <p>Feedback: {response.feedback}</p>
                </div>
            )}
        </div>
    );
};

export default Chatbot;
