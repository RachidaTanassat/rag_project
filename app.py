import streamlit as st
import os
import shutil
import subprocess
import requests

DATA_PATH = "data"
PREPARE_DATA_SCRIPT = "prepare_data.py"

def main():
    st.title("🦜🔗 RAG App With LangChain")

    st.sidebar.header("Settings")
    
    uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type="pdf")
        
    if uploaded_file is not None:
        # Remove the existing DATA_PATH directory if it exists
        if os.path.exists(DATA_PATH):
            shutil.rmtree(DATA_PATH)
        
        # Create a new DATA_PATH directory
        os.makedirs(DATA_PATH)
        
        # Save the uploaded file to the DATA_PATH directory
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write(f"Uploaded file: {uploaded_file.name}")
        
        # Run the data preparation script
        st.write("Processing your PDF...")
        result = subprocess.run(
            ["python", PREPARE_DATA_SCRIPT],
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
    # Replace with the actual endpoint URL
    response = requests.post("http://localhost:5000/query", json={"query": query_text})
    
    if response.status_code == 200:
        return response.json().get("response", "No response found.")
    else:
        return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    main()
