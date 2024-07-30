from flask import Flask, request, jsonify
import os
import dotenv
from groq import Groq
from langchain.prompts import ChatPromptTemplate
from similarity import search_similar_documents

dotenv.load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

----
Answer the question based on the above context: {question}
"""

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    context_text = search_similar_documents(query_text)
    if not context_text:
        return jsonify({"error": "No results found or no matching results"}), 404
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        response_text = chat_completion.choices[0].message.content
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
