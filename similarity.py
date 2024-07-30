import os
import argparse
import dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
dotenv.load_dotenv()

CHROMA_PATH = './chroma_db'
inference_api_key = os.environ.get("HUGGING_FACE_API_KEY")

def search_similar_documents(query_text, k=3):
    # Initialize the embedding function with the Hugging Face Inference API
    embedding_function = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key,
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    
    # Initialize the Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    
    # Perform similarity search with relevance scores
    results = db.similarity_search_with_relevance_scores(query_text, k)
    if not results:
        return None

    # Filter results based on the threshold
    filtered_results = [(doc, score) for doc, score in results ]
    if not filtered_results:
        return None
    
    # Concatenate the filtered results into a single context text
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    return context_text

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('query_text', type=str, help='Query text.')
    args = parser.parse_args()
    query_text = args.query_text
    
    # Search for similar documents and print the context
    context_text = search_similar_documents(query_text)
    if context_text:
        print(context_text)
    else:
        print("No results found or no matching results.")




# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],
#     model="llama3-8b-8192",
# )

# print(chat_completion.choices[0].message.content)


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