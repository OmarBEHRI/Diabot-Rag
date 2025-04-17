document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-btn');

    // Initialize the chat interface
    initChat();

    function initChat() {
        // Set up event listeners
        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Focus on input field
        userInput.focus();
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input field
        userInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send message to backend and get response
        processUserMessage(message);
    }

    function addMessageToChat(sender, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const paragraph = document.createElement('p');
        paragraph.textContent = content;
        
        messageContent.appendChild(paragraph);
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom of chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot';
        typingDiv.id = 'typing-indicator';
        
        const typingContent = document.createElement('div');
        typingContent.className = 'message-content typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingContent.appendChild(dot);
        }
        
        typingDiv.appendChild(typingContent);
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    async function processUserMessage(message) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error('Failed to get response from server');
            }

            const data = await response.json();
            
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add bot response to chat
            addMessageToChat('bot', data.response);
            
        } catch (error) {
            console.error('Error:', error);
            removeTypingIndicator();
            addMessageToChat('bot', 'Sorry, I encountered an error. Please try again later.');
        }
    }
});