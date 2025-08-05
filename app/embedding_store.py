"""
embedding_store.py

Simple job embeddings creation and storage using ChromaDB with LangChain.
"""

import pandas as pd
import os
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

def create_job_embeddings(csv_path="data/jobs_minimal.csv", 
                         persist_dir="chroma_db",
                         collection_name="job_postings"):
    """
    Create embeddings from job CSV and store in ChromaDB
    
    Args:
        csv_path: Path to CSV file
        persist_dir: Directory to store ChromaDB
        collection_name: Name of the collection
    """
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    csv_full_path = project_root / csv_path
    persist_full_path = project_root / persist_dir
    
    # Load CSV
    df = pd.read_csv(csv_full_path)
    print(f"Loaded {len(df)} job records")
    
    # Create documents
    documents = []
    for _, row in df.iterrows():
        # Combine title_normalized and description
        title = str(row.get('title_normalized', '')) if pd.notna(row.get('title_normalized')) else ""
        description = str(row.get('description', '')) if pd.notna(row.get('description')) else ""
        content = f"{title}. {description}".strip()
        
        if len(content) < 10:  # Skip empty content
            continue
            
        # Create metadata
        metadata = {
            "job_id": int(row['id']) if pd.notna(row.get('id')) else None,
            "title": str(row.get('title', '')) if pd.notna(row.get('title')) else "",
            "company": str(row.get('company', '')) if pd.notna(row.get('company')) else "",
            "location": str(row.get('location', '')) if pd.notna(row.get('location')) else "",
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"Created {len(documents)} documents")
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create and persist vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_full_path),
        collection_name=collection_name
    )
    
    print(f"Embeddings created and saved to {persist_full_path}")
    print(f"Collection: {collection_name}")
    
    return vectorstore

if __name__ == "__main__":
    create_job_embeddings()

# import os
# import pandas as pd
# from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.docstore.document import Document
# import tiktoken

# load_dotenv()

# # === Configuration ===
# CSV_PATH = "data/jobs_minimal.csv"
# CHROMA_PERSIST_DIR = "chroma_db"
# CHROMA_COLLECTION_NAME = "job_embeddings"
# MAX_TOKENS = 8191  # Max for OpenAI embedding models
# EMBEDDING_MODEL = "text-embedding-3-small"

# # === Tokenizer ===
# encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)

# # === Embedding model ===
# embedding_model = OpenAIEmbeddings(
#     model=EMBEDDING_MODEL,
#     openai_api_key=os.getenv("OPENAI_API_KEY")
# )

# def load_jobs(csv_path=CSV_PATH):
#     df = pd.read_csv(csv_path)
#     df.dropna(subset=["title_normalized", "description"], inplace=True)
#     return df

# def num_tokens(text: str) -> int:
#     return len(encoding.encode(text))

# def prepare_documents(df: pd.DataFrame) -> list[Document]:
#     docs = []
#     for _, row in df.iterrows():
#         content = f"{row['title_normalized']} {row['description']}"
#         if num_tokens(content) <= MAX_TOKENS:
#             docs.append(Document(page_content=content, metadata=row.to_dict()))
#         else:
#             print(f"⚠️ Skipping long entry ({num_tokens(content)} tokens): {row['normalized_title'][:40]}...")
#     return docs

# def build_chroma_index():
#     df = load_jobs()
#     documents = prepare_documents(df)
#     print(f"📦 Indexing {len(documents)} documents into ChromaDB...")

#     vectorstore = Chroma.from_documents(
#         documents=documents,
#         embedding=embedding_model,
#         persist_directory=CHROMA_PERSIST_DIR,
#         collection_name=CHROMA_COLLECTION_NAME,
#     )

#     vectorstore.persist()
#     print(f"✅ Stored {len(documents)} jobs in ChromaDB with metadata.")

# if __name__ == "__main__":
#     build_chroma_index()
