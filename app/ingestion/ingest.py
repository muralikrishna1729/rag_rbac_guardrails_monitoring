import os
from pathlib import Path
from langchain_community.document_loaders import CSVLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(data_path:str):
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

    for doc in all_docs:
        source_path = doc.metadata.get("source", "")
        if source_path:
            department_name = Path(source_path).parent.name
            doc.metadata['role'] = department_name

    print(f"Success: Loaded {len(all_docs)} total documents.")

    return all_docs

def split_documents(all_docs:list):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_documents(all_docs)
    print(f"Success: Split into {len(chunks)} chunks.")
    return chunks



if __name__ == "__main__":
    data_path = "./resources"
    print(f"Starting load from: {data_path}...")
    
    results = load_documents(data_path)
    print("\n--- RESULTS ---")
    print(f"Total documents loaded: {len(results)}")
    
    if results:
        for result in results:
            print(f"Sample Metadata from first doc: {result.metadata}")
            print(f"Sample Content: {result.page_content[:20]}...")
