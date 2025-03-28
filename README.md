# RAG Application

## Overview
This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them through a chat interface. The application leverages cutting-edge technologies including LangChain, Streamlit, Graq (LLaMA3), and Hugging Face API to provide accurate and contextually relevant responses.

## Features
- **PDF Upload**: Users can upload PDF documents for analysis.
- **Chat Interface**: Users can interact with the uploaded PDFs through a chat interface.
- **Accurate Retrieval**: Utilizes Hugging Face API for embedding functions to retrieve relevant information.
- **Coherent Responses**: Powered by Graq (LLaMA3) for generating coherent and contextually accurate responses.

## Technologies Used
- **LangChain**: Framework for linking the various components of the RAG system.
- **Streamlit**: User-friendly frontend interface.
- **Graq (LLaMA3)**: Text generation model.
- **Hugging Face API**: Embedding functions for accurate information retrieval.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/RachidaTanassat/rag_project.git
    ```


2. **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Configuration
Ensure you have the following environment variables set up:

- `HUGGING_FACE_API_KEY`: Your Hugging Face API key.
- `GRAQ_API_KEY`: Your Graq (LLaMA3) API key.

You can set these environment variables in a `.env` file at the root of your project.

## Technical Schema
<img width="776" alt="schema" src="https://github.com/user-attachments/assets/2ffecc84-388e-4e6a-ab46-8aea8b5560c8">

## Demo video


https://github.com/user-attachments/assets/53276e68-e1aa-4d97-8fed-2bdad6d3e88b






