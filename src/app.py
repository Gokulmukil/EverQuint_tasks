# app.py
from src.data_loader import load_all_documents

if __name__ == "__main__":
    # You must provide the path to your data folder (e.g., "data/")
    # If your PDFs are in a folder named 'data', use this:
    docs = load_all_documents("data/") 

    print(f"Loaded {len(docs)} documents.")
    print(docs)