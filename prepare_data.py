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

def main():
    generate_data_store()

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
    
    document = chunks[1]
    print(document.page_content)
    print(document.metadata)
    return chunks

# Initialize the embedding function using Hugging Face Inference API
inference_api_key = os.environ.get("HUGGING_FACE_API_KEY")
embedding_function = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

def save_to_chroma(chunks: list[Document]):
    # Remove the existing Chroma database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Initialize the Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    
    # Add documents to the database
    db.add_documents(chunks)
    print(f"Added {len(chunks)} new chunks to the database.")
    
if __name__ == '__main__':
    main()

    
# def save_to_chroma(chunks: list[Document]):
#     db =  Chroma.from_documents(
#         chunks,
#         persist_directory=CHROMA_PATH,
#         embedding_function=get_embedding_function()
#     )
#     #add or update the documents
#     existing_items = db.get(include=[])
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in the database:{len(existing_ids)}")
#     new_chunks = []
#     for chunk in existing_ids:
#         if chunk.metadata["id"] not in existing_ids:
#             new_chunks.append(chunk)
#     new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]       
#     db.add_documents(new_chunks, ids = new_chunks_ids)
#     db.persist()









