import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.0)
    prompt_template = PromptTemplate(
        input_variables=['context', 'question'],
        template="Answer based on context. Question: {question}\nContext:\n{context}\nIf unknown, say 'Answer not available'."
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("faiss_index"):
        st.error("No PDF data found. Please upload and process PDFs first.")
        return
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with multiple PDF")
    st.header("Chat with multiple PDF")
    
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF files and click Submit", accept_multiple_files=True, type="pdf")
        if st.button("Submit and Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete!")

    user_question = st.text_input("Ask a question from the PDF files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
