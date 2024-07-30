import os
import dotenv
from groq import Groq
from langchain.prompts import ChatPromptTemplate
from similarity import search_similar_documents

dotenv.load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

----
Answer the question based on the above context: {question}
"""

def main():
    query_text = input("Enter your query: ")
    
    context_text = search_similar_documents(query_text)
    if not context_text:
        print("No results found or no matching results.")
        return
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
    )
    
    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()
