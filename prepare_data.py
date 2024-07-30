from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
import shutil
import dotenv

dotenv.load_dotenv()

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def generate_data_store():
    documents = load_documents()
    chunks = split_documents(documents)
    save_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Remove the existing Chroma database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Initialize the Chroma database
    inference_api_key = os.environ.get("HUGGING_FACE_API_KEY")
    embedding_function = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key,
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    
    # Add documents to the database
    db.add_documents(chunks)
    print(f"Added {len(chunks)} new chunks to the database.")

if __name__ == '__main__':
    generate_data_store()
