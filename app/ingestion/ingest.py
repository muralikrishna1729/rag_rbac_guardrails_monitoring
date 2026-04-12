import os
from pathlib import Path
from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(data_path:str):
    """Loading: DirectoryLoader pulls the raw text."""
    all_docs = []
    text_loader = DirectoryLoader(
        data_path,
        glob = "**/*.md",
        loader_cls=  TextLoader
    )
    csv_loader = DirectoryLoader(
        data_path,
        glob = "**/*.csv",
        loader_cls = CSVLoader
    )
    all_docs.extend(text_loader.load())
    all_docs.extend(csv_loader.load())

    """
    Tagging:adds the role (department) to the metadata.
    """
    for doc in all_docs:
        source_path = doc.metadata.get("source", "")
        if source_path:
            department_name = Path(source_path).parent.name
            doc.metadata['role'] = department_name

    print(f"Success: Loaded {len(all_docs)} total documents.")

    return all_docs

def split_documents(all_docs:list):
    """
    Splitting: RecursiveCharacterTextSplitter breaks long docs into 1000-character chunks (while keeping that role tag on every chunk).

    """
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators=["\n\n", "\n", ".", " ", ""])
    chunks = splitter.split_documents(all_docs)
    print(f"Success: Split into {len(chunks)} chunks.")
    return chunks

def store_in_chroma(chunks:list,persist_directory:str):
    """
    Embedding: HuggingFaceEmbeddings converts the text into numbers (vectors).

    """
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    """
    Storage: Chroma saves the vectors and metadata to disk.
    """
    vector_db = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_directory
    )
    print(f"Success: Stored chunks in ChromaDB at {persist_directory}")
    return vector_db


if __name__ == "__main__":
    data_path = "./resources/data"
    DB_path = "./chroma_db"
    print(f"Starting load from: {data_path}...")
        
    raw_docs = load_documents(data_path) 
    print(f"Total documents loaded: {len(raw_docs)}")
    
    if raw_docs:
        doc_chunks = split_documents(raw_docs)
        vector_store = store_in_chroma(doc_chunks,DB_path)
        print("\n--- INGESTION COMPLETE ---")
        # Verify the role metadata exists in a sample chunk
        print(f"Sample Chunk Metadata: {doc_chunks[0].metadata}")

