// script.js

document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
    const conversationList = document.getElementById('conversation-list');
    const endConversationButton = document.getElementById('end-conversation');

    let conversations = [];
    let currentConversation = [];

    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = userInput.value.trim();
        if (message === '') {
            return;
        }
        appendMessage('You', message, 'user-message');
        currentConversation.push({sender: 'You', message: message});
        userInput.value = '';

        fetch('/get_response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({'message': message})
        })
        .then(response => response.json())
        .then(data => {
            appendMessage('Bot', data.message, 'bot-message');
            currentConversation.push({sender: 'Bot', message: data.message});
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    function appendMessage(sender, message, className) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', className);
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatHistory.appendChild(messageElement);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function saveConversation() {
        if (currentConversation.length > 0) {
            conversations.push(currentConversation);
            updateConversationList();
            currentConversation = [];
            chatHistory.innerHTML = '';
        }
    }

    function updateConversationList() {
        conversationList.innerHTML = '';
        conversations.forEach((conv, index) => {
            const convItem = document.createElement('li');
            convItem.textContent = `Conversation ${index + 1}`;
            convItem.addEventListener('click', () => {
                loadConversation(index);
            });
            conversationList.appendChild(convItem);
        });
    }

    function loadConversation(index) {
        chatHistory.innerHTML = '';
        const conv = conversations[index];
        conv.forEach(msg => {
            appendMessage(msg.sender, msg.message, msg.sender === 'You' ? 'user-message' : 'bot-message');
        });
    }

    endConversationButton.addEventListener('click', saveConversation);
});
