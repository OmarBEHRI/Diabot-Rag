# DiaBot 🤖💉 - Your Friendly Neighborhood Diabetes AI Assistant!

[![Chat Interface](placeholder_for_screenshot.png)](http://localhost:3000) <!-- Optional: Add a screenshot later -->

Welcome to DiaBot! Ever wished you had a knowledgeable buddy to answer your diabetes questions quickly and accurately? Well, wish no more! DiaBot is here to help, powered by some cool AI magic and a brain full of diabetes-specific info.

This isn't just any chatbot. DiaBot uses a fancy technique called **Retrieval Augmented Generation (RAG)**. Think of it like an open-book exam for the AI – instead of just guessing, it first looks up relevant information from its special diabetes knowledge base *before* answering your question. This means you get answers that are grounded in the data we provide it!

## Features That Make DiaBot Awesome ✨

- **💬 Snazzy Chat Interface:** A clean, diabetes-themed web interface to chat with the bot.
- **🧠 Smart Answers (RAG):** Doesn't just make things up! Retrieves relevant info from its knowledge base first.
- **🚀 Powered by Gemini `1.5-flash-latest`:** Uses Google's speedy and capable AI model to understand you and generate helpful responses.
- **🏠 Local Embeddings Power!:** Uses the `sentence-transformers/all-MiniLM-L6-v2` model running *directly on your machine* (using your GPU if you have one!) to understand text meaning. No Hugging Face API key needed for this part!
- **📚 Specialized Diabetes Knowledge:** You feed it information specifically about diabetes (in the `data` folder).
- **🌐 Web-Based:** Access it from your browser on your computer.
- **🐍 Built with Python & FastAPI:** Modern, fast, and efficient backend.

## How Does The Magic Happen? 🎩🐇

DiaBot combines several cool pieces of tech:

1. **The Knowledge Base (`data/` folder):** This is where you put all the trustworthy diabetes information (as `.txt` files). This is DiaBot's "textbook".
2. **Embeddings (The Meaning Mapper):** When the app first starts (or if the `vectorstore` folder is missing), it reads all the text in your `data` folder. It uses the local `sentence-transformers/all-MiniLM-L6-v2` model (running on your CPU or GPU thanks to PyTorch!) to turn each chunk of text into a list of numbers (a "vector"). These vectors capture the *meaning* of the text. Think of it as plotting each piece of info on a giant "meaning map".
3. **Vector Store (The Super-Fast Library - FAISS):** All those numerical vectors are stored in a highly efficient index using FAISS (in the `vectorstore` folder). This allows the bot to *very quickly* find text chunks whose meanings are similar to your question.
4. **User Question:** You type your question into the chat.
5. **Query Embedding:** Your question *also* gets turned into a numerical vector using the *same* local embedding model.
6. **Retrieval (Finding the Clues):** DiaBot takes your question's vector and uses FAISS to instantly find the text chunks from the knowledge base whose vectors are mathematically closest (most similar in meaning).
7. **Augmented Prompt (Giving the AI Context):** DiaBot creates a special prompt for the Gemini AI. This prompt includes:
    - The relevant text chunks it just retrieved.
    - Your original question.
    - Instructions telling it to answer based *only* on the provided context if possible.
8. **Generation (The Grand Finale - Gemini `2.0-flash`):** This augmented prompt is sent to the Google Gemini API (`gemini-2.0-flash` model). Gemini reads the context and your question and generates a helpful, context-aware answer.
9. **Display:** The answer pops up in your chat window!

## Tech Stack 🥞

- **Backend:** Python 3.8+
- **Web Framework:** FastAPI
- **Web Server:** Uvicorn
- **AI Orchestration:** LangChain
- **LLM:** Google Gemini API (`gemini-2.0-flash`)
- **Local Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`) via `langchain_community.embeddings.HuggingFaceEmbeddings`
- **Machine Learning Lib:** PyTorch (for running embeddings locally, especially on GPU)
- **Vector Store:** FAISS (CPU version `faiss-cpu`, but embeddings can still use GPU)
- **Frontend:** HTML, CSS, JavaScript

## Get Ready to Launch! 🚀 (Setup & Installation)

Okay, let's get DiaBot running on your machine. Follow these steps carefully!

### 1. Prerequisites:

- **Python:** You need Python installed. Version 3.8 or higher is recommended. You can check by opening a terminal/command prompt and typing `python --version`. Download from [python.org](https://www.python.org/) if needed.
- **Pip:** Python's package installer. Usually comes with Python. Check with `pip --version`.
- **Git:** Needed to clone the repository (if you haven't already). ([https://git-scm.com/downloads](https://git-scm.com/downloads))
- **(Optional but HIGHLY Recommended) NVIDIA GPU:** If you want the embedding process to be *much* faster, an NVIDIA GPU (like your RTX 4060) is great. Make sure you have the latest NVIDIA drivers installed.

### 2. Get the Code:

If you haven't already, clone the repository or download the source code into a folder (e.g., `DiaBot Version 1`). Open your terminal or command prompt and navigate into that folder:

```bash
# If cloning:
# git clone <your-repository-url>
cd "DiaBot Version 1"
```

### 3. Set Up a Virtual Environment (Highly Recommended!):

This keeps the project's Python packages separate from your system's Python.

```bash
python -m venv venv
```

Now, activate the virtual environment. The command differs based on your OS/terminal:

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

(If you get an error about execution policies, you might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, just for this terminal session).

**Windows (Command Prompt - cmd.exe):**

```cmd
venv\Scripts\activate.bat
```

**Linux / macOS (Bash/Zsh):**

```bash
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your terminal prompt. Keep this terminal open and activated for the next steps.

### 4. Install PyTorch with CUDA (For GPU Power!):

This is the most important step for using your GPU. Do not just rely on `requirements.txt` for this.

Go to the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Select the options that match your system:

- **PyTorch Build:** Stable
- **Your OS:** Windows (or Linux/macOS if applicable)
- **Package:** Pip
- **Language:** Python
- **Compute Platform:** CUDA (select the version compatible with your drivers, e.g., CUDA 11.8 or CUDA 12.1). If unsure, the latest listed CUDA version is usually fine if your drivers are up-to-date.

Copy the generated `pip install ...` command. It will look something like this (DO NOT COPY THIS EXAMPLE, GET YOURS FROM THE SITE):

```bash
# Example Only! Get the command from the PyTorch website!
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Paste and run that specific command in your activated virtual environment terminal. This might take a few minutes to download and install.

### 5. Install Other Requirements:

Now that PyTorch is correctly installed, install the rest of the packages:

```bash
pip install -r requirements.txt
```

### 6. Create and Configure the `.env` File:

This file holds your secret API key. It's not included in the repository for security.

Create a new file named exactly `.env` in the main project directory (`DiaBot Version 1/`).

Open the `.env` file in a text editor.

Add the following line, replacing `YOUR_KEY_HERE` with your actual Gemini API key:

```dotenv
GEMINI_API_KEY=YOUR_KEY_HERE
```

**How to get a Gemini API Key?** Go to Google AI Studio and create one. Keep it secret!

**IMPORTANT:** Make sure `.env` is listed in your `.gitignore` file if you're using Git, so you don't accidentally commit your key!

### 7. Add Your Diabetes Data:

Make sure the `data` directory exists.

Place your `.txt` files containing diabetes information inside the `data` directory. The more relevant info you add, the better DiaBot can potentially answer!

Phew! Setup complete! 🎉

## Running DiaBot 🏃‍♀️💨

Make sure your virtual environment is still activated (you should see `(venv)` in your prompt).

Navigate to the project directory (`DiaBot Version 1/`) in your terminal if you aren't already there.

Run the Uvicorn server:

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 3000 --reload
```

- `server:app`: Tells Uvicorn to find the `app` object inside the `server.py` file.
- `--host 127.0.0.1`: Makes the server accessible only from your own computer (localhost). Use `--host 0.0.0.0` if you need to access it from other devices on your local network.
- `--port 3000`: The port number the server will listen on.
- `--reload`: Automatically restarts the server if you make changes to `server.py` (great for development!).

**First Run Patience:** The very first time you run this, it needs to:

- Download the embedding model (`all-MiniLM-L6-v2`).
- Read all your `.txt` files from `data/`.
- Generate embeddings for all the text chunks (this uses your GPU if PyTorch was set up correctly, otherwise CPU, and can take a while depending on data size and hardware!).
- Create the FAISS index and save it to the `vectorstore/` directory.

You'll see log messages in the terminal indicating progress. Subsequent runs will be much faster as they'll load the existing `vectorstore`.

### Access the Chatbot:

Once you see `INFO: Uvicorn running on http://127.0.0.1:3000` (or similar), open your web browser and go to:

```
http://localhost:3000
```

## Using DiaBot 🗣️

It's super simple:

1. The chat interface will load in your browser.
2. Type your diabetes-related question into the input box at the bottom.
3. Press Enter or click the send button (✈️).
4. Watch DiaBot think (you'll see a typing indicator) and then deliver its RAG-powered answer!

## Customization Ideas 🎨

- **Styling:** Modify `styles.css` to change the look and feel.
- **Knowledge:** Add more `.txt` files to the `data/` directory to expand DiaBot's knowledge. Remember to delete the `vectorstore` folder and restart the server to rebuild the index after adding/changing data significantly.
- **Models:** Experiment with different Sentence Transformer embedding models from Hugging Face (in `server.py`, change `EMBEDDING_MODEL_NAME`) or different Gemini models (change `model="..."` in `server.py`). Note that different models have different resource requirements and performance.
- **Prompt:** Edit the system prompt within the `chat_endpoint` function in `server.py` to change DiaBot's personality or instructions.
