import streamlit as st
import os
import dotenv
import subprocess
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
    st.title("🦜🔗 RAG App With LangChain")

    st.sidebar.header("Settings")
    
    uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type="pdf")
        
    if uploaded_file is not None:
        st.write(f"Uploaded file: {uploaded_file.name}")
        
        # Process the PDF file (you can add your PDF processing logic here)
        st.write("Processing your PDF...")
        # Assuming `prepare_data.py` processes the PDF and saves necessary data
        result = subprocess.run(
            ["python", "prepare_data.py"],
            capture_output=True,
            text=True
        )
        
        st.success("Data preparation complete!")

    st.header("Chat with Bot 🤖")
    
    # Initialize chat history and text input state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    # Form for chat input and display
    with st.form(key="my_form"):
        # Display chat history within the form
        for message in st.session_state.chat_history:
            if message['role'] == 'bot':
                st.markdown(
                    f"""
                    <div style='text-align: left; padding: 10px; border-radius: 8px; background-color: #262730; color: #ffffff; margin-bottom: 20px; margin-top: 60px; display: inline-block;'>
                        🤖 {message['text']}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            elif message['role'] == 'user':
                st.markdown(
                    f"""
                    <div style='text-align: left; right: 0; position: absolute; padding: 10px; border-radius: 8px; background-color: #262730; color: #ffffff; margin-bottom: 60px; margin-top: 20px; display: inline-block;'>
                        {message['text']}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

        # Input field and submit button
        query_text = st.text_area("Enter your query:")
        submit_button = st.form_submit_button("Send")
        
        if not uploaded_file:
            st.warning("Please upload your file!", icon="⚠")

        if submit_button and query_text:
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'text': query_text
            })
            
            # Get bot response
            response = query_data(query_text)
            
            # Add bot response to chat history
            st.session_state.chat_history.append({
                'role': 'bot',
                'text': response
            })
            
            # Clear the input field
            st.experimental_rerun()

def query_data(query_text):
    context_text = search_similar_documents(query_text)
    if not context_text:
        return "No results found or no matching results"

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        response_text = chat_completion.choices[0].message.content
        return response_text
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    main()
