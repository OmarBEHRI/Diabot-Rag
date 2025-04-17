# DiaBot - AI-powered Diabetes Assistant

DiaBot is a web-based chatbot that uses Retrieval Augmented Generation (RAG) to provide accurate and helpful information about diabetes. It combines the power of Google's Gemini AI with a specialized diabetes knowledge base to deliver personalized responses to user queries.

## Features

- 💬 Interactive chat interface with a diabetes-themed design
- 🔍 Retrieval Augmented Generation for accurate diabetes information
- 🧠 Powered by Google's Gemini AI model
- 📚 Specialized diabetes knowledge base
- 🌐 Web-based interface accessible from any device

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Node.js, Express
- **AI**: Google Gemini API
- **Embeddings**: Hugging Face Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS for efficient similarity search

## Setup Instructions

### Prerequisites

- Node.js (v14 or higher)
- NPM (v6 or higher)
- Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Hugging Face API key from [Hugging Face](https://huggingface.co/settings/tokens)

### Installation

1. Clone the repository or download the source code

2. Install dependencies
   ```
   npm install
   ```

3. Create a `.env` file based on the provided `.env.example`
   ```
   cp .env.example .env
   ```

4. Add your API keys to the `.env` file
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   HF_API_KEY=your_huggingface_api_key_here
   ```

5. Create a `data` directory and add diabetes information text files
   ```
   mkdir data
   ```
   Add `.txt` files containing diabetes information to this directory.

6. Start the server
   ```
   npm start
   ```

7. Open your browser and navigate to `http://localhost:3000`

## Usage

1. Type your diabetes-related question in the chat input field
2. Press Enter or click the send button
3. DiaBot will process your question and provide a helpful response based on its diabetes knowledge base

## Customization

- Modify the `styles.css` file to change the appearance of the chatbot
- Add more diabetes information to the `data` directory to expand the knowledge base
- Adjust the RAG parameters in `server.js` to fine-tune the retrieval process

## License

MIT