import os
import logging
from dotenv import load_dotenv
import torch # PyTorch is needed for sentence-transformers

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse # To serve index.html
from pydantic import BaseModel # For request body validation

# Langchain components
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader # Using community loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Using local HF embeddings
from langchain_google_genai import GoogleGenerativeAI # Updated import

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Embedding model config - Runs locally
# Use cuda if available, otherwise cpu. Your RTX 4060 should be detected if PyTorch is installed correctly.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")
if DEVICE == "cpu":
     logging.warning("CUDA not available or PyTorch not installed with CUDA support. Running embeddings on CPU.")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Good starting point
# Larger models might give better results but need more VRAM/RAM
# e.g., "sentence-transformers/all-mpnet-base-v2"
# e.g., "BAAI/bge-small-en-v1.5"

DATA_DIR = "data"
VECTORSTORE_PATH = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Initialization ---
app = FastAPI()

# Serve static files (like index.html, css, js) from the root directory
# Mount this *before* the catch-all route if you have one
app.mount("/static", StaticFiles(directory="."), name="static")


# Global variable for vector store (can be improved with FastAPI dependency injection later)
vector_store = None

# Initialize Embeddings Model (runs locally on CPU or GPU)
try:
    logging.info(f"Initializing HuggingFaceEmbeddings model: {EMBEDDING_MODEL_NAME} on device: {DEVICE}")
    # Specify encode_kwargs to normalize embeddings for potentially better FAISS performance
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    logging.info("HuggingFace Embeddings model initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing HuggingFace Embeddings model: {e}", exc_info=True)
    # Decide if you want the app to fail or continue without RAG
    raise RuntimeError("Failed to load embedding model.") from e


# Initialize Gemini API
try:
    logging.info("Initializing Google Generative AI client...")
    genai_model = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
    logging.info("Google Generative AI client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Google Generative AI client: {e}", exc_info=True)
    raise RuntimeError("Failed to initialize Gemini client.") from e


# --- Helper Functions ---
def initialize_or_load_vectorstore():
    """Loads vector store if it exists, otherwise creates it."""
    global vector_store
    faiss_index_path = os.path.join(VECTORSTORE_PATH, "index.faiss")
    faiss_pkl_path = os.path.join(VECTORSTORE_PATH, "index.pkl")

    if os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path):
        try:
            logging.info(f"Loading existing vector store from: {VECTORSTORE_PATH}")
            # Allow dangerous deserialization if you trust the source of the index files
            vector_store = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            logging.info("Vector store loaded successfully.")
            return True
        except Exception as e:
            logging.error(f"Error loading existing vector store: {e}", exc_info=True)
            logging.warning("Failed to load existing vector store, will attempt to recreate.")
            # Clear potentially corrupted files if loading failed
            if os.path.exists(faiss_index_path): os.remove(faiss_index_path)
            if os.path.exists(faiss_pkl_path): os.remove(faiss_pkl_path)

    logging.info("No existing vector store found or loading failed. Initializing knowledge base...")
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
         logging.error(f"Data directory '{DATA_DIR}' is empty or does not exist. Cannot build vector store.")
         return False

    try:
        # Load documents (only .txt files in this example)
        # Use DirectoryLoader for simplicity if all files are of the same type
        # For mixed types, load manually or use UnstructuredDirectoryLoader (requires more deps)
        loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, recursive=True, show_progress=True)
        documents = loader.load()

        if not documents:
            logging.error(f"No documents loaded from '{DATA_DIR}'. Check file patterns and content.")
            return False

        logging.info(f"Loaded {len(documents)} documents.")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(split_docs)} document chunks.")

        if not split_docs:
             logging.error("Text splitting resulted in zero chunks. Check chunk size/overlap and document content.")
             return False

        # Create vector store using FAISS
        logging.info("Creating FAISS vector store... (This might take a while depending on data size and hardware)")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        logging.info("Vector store created successfully.")

        # Save the vector store for future use
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
        vector_store.save_local(VECTORSTORE_PATH)
        logging.info(f"Vector store saved to: {VECTORSTORE_PATH}")
        return True

    except Exception as e:
        logging.error(f"Error initializing knowledge base: {e}", exc_info=True)
        vector_store = None # Ensure vector_store is None if initialization fails
        return False

# --- Pydantic Models for Request/Response ---
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# --- API Endpoints ---

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Assuming your HTML file is named index.html and is in the same directory as server.py
    # Adjust path if necessary
    index_path = os.path.join(os.path.dirname(__file__), 'index.html')
    if not os.path.exists(index_path):
        return HTMLResponse(content="<html><body><h1>Chatbot UI not found (index.html)</h1></body></html>", status_code=404)

    with open(index_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Handles incoming chat messages, performs RAG, and returns AI response."""
    user_message = chat_message.message
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    context = ""
    if vector_store:
        try:
            logging.info(f"Performing similarity search for: '{user_message[:50]}...'")
            # Retrieve relevant context (e.g., top 3 chunks)
            results = vector_store.similarity_search(user_message, k=3)
            context = "\n\n".join([doc.page_content for doc in results])
            logging.info(f"Retrieved {len(results)} context chunks.")
            # Optional: Log retrieved context for debugging (can be verbose)
            # logging.debug(f"Retrieved context:\n{context}")

        except Exception as e:
            logging.warning(f"Error retrieving from vector store: {e}", exc_info=True)
            # Continue without context if vector store search fails
            context = "Vector store search failed." # Provide feedback in context
    else:
        logging.warning("Vector store not initialized, proceeding without RAG context.")
        context = "Knowledge base context is unavailable." # Provide feedback


    # --- Prepare Prompt for LLM ---
    # Note: Adjust this prompt template based on Gemini's best practices
    # and the desired personality/role of your chatbot.
    prompt = f"""You are DiaBot, a helpful and friendly AI assistant specialized in diabetes information and support. Your knowledge comes from a specific set of documents.

Based ONLY on the following relevant information from the diabetes knowledge base, answer the user's query. If the provided context does not contain the answer, state clearly that the information is not available in the knowledge base. Do not provide general diabetes information unless it is directly supported by the context below. Always be supportive and encouraging.

Relevant information:
--- Start of Context ---
{context}
--- End of Context ---

User query: {user_message}

Response:"""

    # --- Generate Response using Gemini ---
    try:
        logging.info("Sending prompt to Gemini...")
        # Optional: Log the full prompt for debugging
        # logging.debug(f"Gemini Prompt:\n{prompt}")

        # Use the invoke method for Langchain integration
        llm_response = genai_model.invoke(prompt)

        # Accessing the response content might vary slightly depending on the Langchain version
        # Check the structure of llm_response if this doesn't work
        response_text = llm_response if isinstance(llm_response, str) else getattr(llm_response, 'content', str(llm_response))

        logging.info("Received response from Gemini.")
        return ChatResponse(response=response_text)

    except Exception as e:
        logging.error(f"Error generating response from Gemini: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get response from AI model.")

# --- Application Startup ---
@app.on_event("startup")
async def startup_event():
    """Initialize vector store when the application starts."""
    logging.info("Application startup: Initializing vector store...")
    initialized = initialize_or_load_vectorstore()
    if not initialized:
        logging.warning("Vector store could not be initialized. RAG features will be disabled.")
    else:
        logging.info("Vector store ready.")

# --- Main Execution ---
# This part is typically used for local development/debugging.
# For production, you'd use a production server like Gunicorn with Uvicorn workers.
# Example: uvicorn server:app --host 0.0.0.0 --port 3000 --reload
if __name__ == "__main__":
    import uvicorn
    # Run initialization directly before starting uvicorn for simplicity here
    # In a larger app, the startup event is preferred
    print("Initializing knowledge base before server start...")
    initialized = initialize_or_load_vectorstore()
    if not initialized:
         print("WARNING: Vector store could not be initialized. RAG features will be disabled.")
    else:
         print("Vector store ready.")

    print("Starting FastAPI server...")
    # Use --reload for development to automatically restart on code changes
    uvicorn.run("server:app", host="127.0.0.1", port=3000, reload=True)