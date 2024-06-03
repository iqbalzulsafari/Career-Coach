document.querySelector('#user-input').addEventListener('keydown', async function(event) {
    if (event.key === 'Enter') {
        const input = event.target.value;
        if (input.trim() !== '') {
            addMessage('user', input);
            event.target.value = '';

            // Send the user's message to the server and get the bot's response
            const botResponse = await getBotResponse(input);
            addMessage('bot', botResponse);
        }
    }
});

function addMessage(sender, text) {
    const message = document.createElement('div');
    message.classList.add('message', sender);
    message.innerHTML = `<p>${text}</p>`;
    document.querySelector('#chat-box').appendChild(message);
    document.querySelector('#chat-box').scrollTop = document.querySelector('#chat-box').scrollHeight;
}

async function getBotResponse(userInput) {
    try {
        const response = await fetch('http://localhost:5000/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: userInput }),
        });
        const data = await response.json();
        return data.response;
    } catch (error) {
        console.error('Error:', error);
        return "Sorry, I couldn't process your request.";
    }
}
