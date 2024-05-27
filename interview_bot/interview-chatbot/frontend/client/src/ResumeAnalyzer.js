import React, { useState } from 'react';
import axios from 'axios';

const ResumeAnalyzer = () => {
    const [file, setFile] = useState(null);
    const [feedback, setFeedback] = useState(null);

    const uploadResume = async () => {
        const formData = new FormData();
        formData.append('file', file);

        const res = await axios.post('/upload_resume', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        setFeedback(res.data);
    };

    return (
        <div>
            <input type="file" onChange={(e) => setFile(e.target.files[0])} />
            <button onClick={uploadResume}>Upload Resume</button>
            {feedback && (
                <div>
                    <h3>Skills:</h3>
                    <ul>
                        {feedback.skills.map((skill, index) => (
                            <li key={index}>{skill}</li>
                        ))}
                    </ul>
                    <h3>Experiences:</h3>
                    <ul>
                        {feedback.experiences.map((experience, index) => (
                            <li key={index}>{experience}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default ResumeAnalyzer;
