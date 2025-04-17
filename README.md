{
  "readme_content": "# DiaBot 🤖💉 - Your Friendly Neighborhood Diabetes AI Assistant!\n\n[![Chat Interface](placeholder_for_screenshot.png)](http://localhost:3000) <!-- Optional: Add a screenshot later -->\n\nWelcome to DiaBot! Ever wished you had a knowledgeable buddy to answer your diabetes questions quickly and accurately? Well, wish no more! DiaBot is here to help, powered by some cool AI magic and a brain full of diabetes-specific info.\n\nThis isn't just any chatbot. DiaBot uses a fancy technique called **Retrieval Augmented Generation (RAG)**. Think of it like an open-book exam for the AI – instead of just guessing, it first looks up relevant information from its special diabetes knowledge base *before* answering your question. This means you get answers that are grounded in the data we provide it!\n\n## Features That Make DiaBot Awesome ✨\n\n*   **💬 Snazzy Chat Interface:** A clean, diabetes-themed web interface to chat with the bot.\n*   **🧠 Smart Answers (RAG):** Doesn't just make things up! Retrieves relevant info from its knowledge base first.\n*   **🚀 Powered by Gemini `1.5-flash-latest`:** Uses Google's speedy and capable AI model to understand you and generate helpful responses.\n*   **🏠 Local Embeddings Power!:** Uses the `sentence-transformers/all-MiniLM-L6-v2` model running *directly on your machine* (using your GPU if you have one!) to understand text meaning. No Hugging Face API key needed for this part!\n*   **📚 Specialized Diabetes Knowledge:** You feed it information specifically about diabetes (in the `data` folder).\n*   **🌐 Web-Based:** Access it from your browser on your computer.\n*   **🐍 Built with Python & FastAPI:** Modern, fast, and efficient backend.\n\n## How Does The Magic Happen? 🎩🐇\n\nDiaBot combines several cool pieces of tech:\n\n1.  **The Knowledge Base (`data/` folder):** This is where you put all the trustworthy diabetes information (as `.txt` files). This is DiaBot's \"textbook\".\n2.  **Embeddings (The Meaning Mapper):** When the app first starts (or if the `vectorstore` folder is missing), it reads all the text in your `data` folder. It uses the local `sentence-transformers/all-MiniLM-L6-v2` model (running on your CPU or GPU thanks to PyTorch!) to turn each chunk of text into a list of numbers (a \"vector\"). These vectors capture the *meaning* of the text. Think of it as plotting each piece of info on a giant \"meaning map\".\n3.  **Vector Store (The Super-Fast Library - FAISS):** All those numerical vectors are stored in a highly efficient index using FAISS (in the `vectorstore` folder). This allows the bot to *very quickly* find text chunks whose meanings are similar to your question.\n4.  **User Question:** You type your question into the chat.\n5.  **Query Embedding:** Your question *also* gets turned into a numerical vector using the *same* local embedding model.\n6.  **Retrieval (Finding the Clues):** DiaBot takes your question's vector and uses FAISS to instantly find the text chunks from the knowledge base whose vectors are mathematically closest (most similar in meaning).\n7.  **Augmented Prompt (Giving the AI Context):** DiaBot creates a special prompt for the Gemini AI. This prompt includes:\n    *   The relevant text chunks it just retrieved.\n    *   Your original question.\n    *   Instructions telling it to answer based *only* on the provided context if possible.\n8.  **Generation (The Grand Finale - Gemini `1.5-flash-latest`):** This augmented prompt is sent to the Google Gemini API (`gemini-1.5-flash-latest` model). Gemini reads the context and your question and generates a helpful, context-aware answer.\n9.  **Display:** The answer pops up in your chat window!\n\n## Tech Stack 🥞\n\n*   **Backend:** Python 3.8+\n*   **Web Framework:** FastAPI\n*   **Web Server:** Uvicorn\n*   **AI Orchestration:** LangChain\n*   **LLM:** Google Gemini API (`gemini-1.5-flash-latest`)\n*   **Local Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`) via `langchain_community.embeddings.HuggingFaceEmbeddings`\n*   **Machine Learning Lib:** PyTorch (for running embeddings locally, especially on GPU)\n*   **Vector Store:** FAISS (CPU version `faiss-cpu`, but embeddings can still use GPU)\n*   **Frontend:** HTML, CSS, JavaScript\n\n## Project Structure 📁\n\n```\nDiaBot Version 1/\n├── data/                  # <--- Put your diabetes .txt files here!\n│   └── example.txt\n├── vectorstore/           # <--- FAISS index is automatically created here\n│   ├── index.faiss\n│   └── index.pkl\n├── .env                   # <--- IMPORTANT: Create this file for your API key (DO NOT COMMIT)\n├── index.html             # Frontend chat UI\n├── styles.css             # CSS for the UI\n├── script.js              # Frontend JavaScript logic\n├── server.py              # The Python FastAPI backend server code\n├── requirements.txt       # List of Python packages needed\n└── README.md              # You are here!\n```\n\n## Get Ready to Launch! 🚀 (Setup & Installation)\n\nOkay, let's get DiaBot running on your machine. Follow these steps carefully!\n\n**1. Prerequisites:**\n\n*   **Python:** You need Python installed. Version 3.8 or higher is recommended. You can check by opening a terminal/command prompt and typing `python --version`. Download from [python.org](https://www.python.org/) if needed.\n*   **Pip:** Python's package installer. Usually comes with Python. Check with `pip --version`.\n*   **Git:** Needed to clone the repository. ([https://git-scm.com/downloads](https://git-scm.com/downloads))\n*   **(Optional but HIGHLY Recommended) NVIDIA GPU:** If you want the embedding process to be *much* faster, an NVIDIA GPU (like your RTX 4060) is great. Make sure you have the latest NVIDIA drivers installed.\n\n**2. Clone the Repository:**\n\nOpen your terminal or command prompt and navigate to where you want to store the project. Then clone it:\n\n```bash\ngit clone <your-repository-url> # Replace with the actual URL if it's on GitHub/GitLab etc.\ncd DiaBot Version 1             # Or whatever you named the folder\n```\n\n**3. Set Up a Virtual Environment (Highly Recommended!):**\n\nThis keeps the project's Python packages separate from your system's Python.\n\n```bash\npython -m venv venv\n```\n\nNow, **activate** the virtual environment. The command differs based on your OS/terminal:\n\n*   **Windows (PowerShell):**\n    ```powershell\n    .\\venv\\Scripts\\Activate.ps1\n    ```\n    (If you get an error about execution policies, you might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, just for this terminal session).\n*   **Windows (Command Prompt - cmd.exe):**\n    ```cmd\n    venv\\Scripts\\activate.bat\n    ```\n*   **Linux / macOS (Bash/Zsh):**\n    ```bash\n    source venv/bin/activate\n    ```\n\n    You should see `(venv)` appear at the beginning of your terminal prompt. *Keep this terminal open and activated for the next steps.*\n\n**4. Install PyTorch with CUDA (For GPU Power!):**\n\nThis is the *most important* step for using your GPU. **Do not just rely on `requirements.txt` for this.**\n\n*   Go to the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)\n*   Select the options that match your system:\n    *   **PyTorch Build:** Stable\n    *   **Your OS:** Windows (or Linux/macOS if applicable)\n    *   **Package:** Pip\n    *   **Language:** Python\n    *   **Compute Platform:** CUDA (select the version compatible with your drivers, e.g., CUDA 11.8 or CUDA 12.1). If unsure, the latest listed CUDA version is usually fine if your drivers are up-to-date.\n*   Copy the generated `pip install ...` command. It will look something like this ( ***DO NOT COPY THIS EXAMPLE, GET YOURS FROM THE SITE*** ):\n    ```bash\n    # Example Only! Get the command from the PyTorch website!\n    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n    ```\n*   Paste and run **that specific command** in your **activated virtual environment** terminal. This might take a few minutes to download and install.\n\n**5. Install Other Requirements:**\n\nNow that PyTorch is correctly installed, install the rest of the packages:\n\n```bash\npip install -r requirements.txt\n```\n\n**6. Create and Configure the `.env` File:**\n\nThis file holds your secret API key. It's *not* included in the repository for security.\n\n*   Create a new file named exactly `.env` in the main project directory (`DiaBot Version 1/`).\n*   Open the `.env` file in a text editor.\n*   Add the following line, replacing `YOUR_KEY_HERE` with your actual Gemini API key:\n\n    ```dotenv\n    GEMINI_API_KEY=YOUR_KEY_HERE\n    ```\n\n*   **How to get a Gemini API Key?** Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and create one. Keep it secret!\n*   **IMPORTANT:** Make sure `.env` is listed in your `.gitignore` file if you're using Git, so you don't accidentally commit your key!\n\n**7. Add Your Diabetes Data:**\n\n*   Make sure the `data` directory exists.\n*   Place your `.txt` files containing diabetes information inside the `data` directory. The more relevant info you add, the better DiaBot can potentially answer!\n\n**Phew! Setup complete!** 🎉\n\n## Running DiaBot 🏃‍♀️💨\n\n1.  Make sure your virtual environment is still **activated** (you should see `(venv)` in your prompt).\n2.  Navigate to the project directory (`DiaBot Version 1/`) in your terminal if you aren't already there.\n3.  Run the Uvicorn server:\n\n    ```bash\n    python -m uvicorn server:app --host 127.0.0.1 --port 3000 --reload\n    ```\n\n    *   `server:app`: Tells Uvicorn to find the `app` object inside the `server.py` file.\n    *   `--host 127.0.0.1`: Makes the server accessible only from your own computer (`localhost`). Use `--host 0.0.0.0` if you need to access it from other devices on your local network.\n    *   `--port 3000`: The port number the server will listen on.\n    *   `--reload`: Automatically restarts the server if you make changes to `server.py` (great for development!).\n\n4.  **First Run Patience:** The very first time you run this, it needs to:\n    *   Download the embedding model (`all-MiniLM-L6-v2`).\n    *   Read all your `.txt` files from `data/`.\n    *   Generate embeddings for all the text chunks (this uses your GPU if PyTorch was set up correctly, otherwise CPU, and can take a while depending on data size and hardware!).\n    *   Create the FAISS index and save it to the `vectorstore/` directory.\n    *   You'll see log messages in the terminal indicating progress. Subsequent runs will be much faster as they'll load the existing `vectorstore`.\n\n5.  **Access the Chatbot:** Once you see `INFO: Uvicorn running on http://127.0.0.1:3000` (or similar), open your web browser and go to:\n\n    [http://localhost:3000](http://localhost:3000)\n\n## Using DiaBot 🗣️\n\nIt's super simple:\n\n1.  The chat interface will load in your browser.\n2.  Type your diabetes-related question into the input box at the bottom.\n3.  Press `Enter` or click the send button (✈️).\n4.  Watch DiaBot think (you'll see a typing indicator) and then deliver its RAG-powered answer!\n\n## Customization Ideas 🎨\n\n*   **Styling:** Modify `styles.css` to change the look and feel.\n*   **Knowledge:** Add more `.txt` files to the `data/` directory to expand DiaBot's knowledge. Remember to delete the `vectorstore` folder and restart the server to rebuild the index after adding/changing data significantly.\n*   **Models:** Experiment with different Sentence Transformer embedding models from Hugging Face (in `server.py`, change `EMBEDDING_MODEL_NAME`) or different Gemini models (change `model=\"...\"` in `server.py`). Note that different models have different resource requirements and performance.\n*   **Prompt:** Edit the system prompt within the `chat_endpoint` function in `server.py` to change DiaBot's personality or instructions.\n\n## License 📜\n\nThis project is likely under the MIT License (based on the original `package.json`). Feel free to add a `LICENSE` file with the MIT license text if you intend to share it widely.\n\n```\n(You can copy the standard MIT License text here if needed)\n```\n\n---\n\nHave fun chatting with DiaBot! If you run into issues, double-check the setup steps, especially the virtual environment activation and the PyTorch installation.\ninstallation.\n"
}