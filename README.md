# RAG (Retrieval-Augmented Generation) System

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline for processing and querying documents. It allows you to load various types of documents like PDFs and text files, convert them into embeddings, store them in a vector database, and then use natural language queries to retrieve relevant information and generate answers using a large language model.

## What This Project Does

The system takes your documents, breaks them down into manageable pieces, creates mathematical representations (embeddings) of the content, and stores them in a way that makes it easy to find relevant information when you ask questions. It then combines this retrieved information with a powerful language model to provide accurate, context-aware answers based on your documents.

## Key Features

- **Multi-format Document Support**: Load and process PDF files, text files, and other document types
- **Intelligent Text Splitting**: Break down large documents into smaller, meaningful chunks for better retrieval
- **Vector Embeddings**: Use state-of-the-art sentence transformers to create semantic representations of text
- **Persistent Vector Storage**: Store embeddings in ChromaDB for fast, persistent retrieval
- **Advanced Retrieval**: Find the most relevant document chunks based on semantic similarity
- **LLM Integration**: Generate natural language answers using Groq's Llama models
- **Flexible Querying**: Support for both simple and advanced RAG queries with confidence scoring

## Project Structure

```
rag/
├── data/
│   ├── pdf/                    # PDF documents for processing
│   └── text_files/             # Text documents
├── notebook/
│   ├── pdf_loader.ipynb        # Main RAG pipeline notebook
│   ├── docuement.ipynb         # Data ingestion examples
│   └── src/                    # Modular code components
│       ├── data_loader.py      # Document loading utilities
│       ├── embedding.py        # Embedding generation
│       ├── search.py           # Search functionality
│       └── vectorestore.py     # Vector database operations
├── src/
│   └── app.py                  # Main application entry point
├── .env                        # Environment variables (API keys)
├── pyproject.toml              # Project configuration
├── requirements.txt            # Python dependencies
├── main.py                     # Alternative entry point
└── README.md                   # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.13 or higher
- A Groq API key (get one from [groq.com](https://groq.com))

### 2. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Prepare Your Data

Place your documents in the `data/` directory:
- PDFs go in `data/pdf/`
- Text files go in `data/text_files/`

The system currently includes sample documents:
- A chemistry textbook PDF
- Python programming introduction text
- Machine learning basics text

## How the RAG Workflow Works

The system follows a clear, step-by-step process to transform your documents into an interactive question-answering system:

### Step 1: Data Ingestion
- Load all documents from the data directory
- Support multiple file formats (PDF, TXT, CSV, etc.)
- Extract text content and preserve metadata like source file names and page numbers
- Handle large documents by processing them page by page

### Step 2: Text Chunking
- Break down long documents into smaller, manageable pieces
- Use intelligent splitting that respects sentence boundaries
- Configure chunk size (default: 1000 characters) and overlap (default: 200 characters)
- This ensures better retrieval accuracy and fits within LLM context limits

### Step 3: Embedding Generation
- Convert text chunks into numerical vectors using sentence transformers
- Use the "all-MiniLM-L6-v2" model for creating 384-dimensional embeddings
- Each piece of text gets transformed into a mathematical representation that captures its semantic meaning
- This allows the system to understand relationships between different pieces of content

### Step 4: Vector Database Storage
- Store embeddings in ChromaDB, a fast and persistent vector database
- Each document chunk is stored with its embedding, original text, and metadata
- The database persists on disk, so you don't need to reprocess documents
- Supports efficient similarity search across thousands of document chunks

### Step 5: Query Processing
- When you ask a question, the system converts it into an embedding
- Searches the vector database for the most similar document chunks
- Returns the top-k most relevant results (configurable, default: 5)
- Filters results based on similarity threshold to ensure quality

### Step 6: Answer Generation
- Combine retrieved document chunks into context
- Send context + your question to a large language model (Llama 3.1 via Groq)
- Generate a natural, coherent answer based on the retrieved information
- Include source information and confidence scores for transparency

## Usage Examples

### Basic Document Loading

```python
from src.data_loader import load_all_documents

# Load all documents from the data directory
docs = load_all_documents("data/")
print(f"Loaded {len(docs)} documents.")
```

### Running the Complete Pipeline

The main pipeline is demonstrated in `notebook/pdf_loader.ipynb`. It shows:
- Loading and processing PDF documents
- Creating embeddings
- Setting up the vector store
- Performing retrieval queries
- Generating answers with the LLM

### Simple Query Example

```python
# After setting up the retriever and LLM
result = rag_simple("What is machine learning?", retriever, llm)
print(result)
```

### Advanced Query with Sources

```python
result = rag_advanced(
    "Explain electronegativity trends",
    retriever,
    llm,
    top_k=3,
    min_score=0.1,
    return_context=True
)

print("Answer:", result["answer"])
print("Sources:", result["sources"])
print("Confidence:", result["confidence"])
```

## Dependencies

The project uses several key libraries:

- **LangChain**: Framework for building LLM applications
- **Sentence Transformers**: For generating text embeddings
- **ChromaDB**: Vector database for storing and retrieving embeddings
- **PyPDF**: PDF text extraction
- **LangChain Groq**: Integration with Groq's LLM API
- **Python-dotenv**: Environment variable management

## Data Format Support

Currently supported document types:
- PDF files (via PyPDFLoader)
- Plain text files (via TextLoader)
- CSV files
- Microsoft Office documents (Word, Excel)
- JSON files

## API Keys and Security

- Store your Groq API key in the `.env` file
- Never commit API keys to version control
- The `.env` file is already in `.gitignore`

## Performance Notes

- Embedding generation can be time-intensive for large document collections
- Vector search is very fast once embeddings are created
- LLM calls have latency based on model size and API response times
- Consider document chunk size for optimal retrieval accuracy

## Future Enhancements

The modular design allows for easy extension:
- Add support for more document types
- Implement different embedding models
- Add web interface for queries
- Support multiple vector databases
- Add document preprocessing and cleaning
- Implement conversation memory for multi-turn queries

This RAG system provides a solid foundation for building document-based question-answering applications with high accuracy and transparency.