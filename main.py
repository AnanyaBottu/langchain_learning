import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from docx import Document

load_dotenv()


def get_file_text(files):
    
    text = ""
    for file in files:
        ext = os.path.splitext(file.name)[1].lower()

        if ext == ".pdf":
            text += read_pdf(file) + "\n"
        elif ext == ".docx":
            text += read_docx(file) + "\n"
        else:
            st.warning(f"Unsupported file format: {ext}")
    return text


def read_pdf(pdf_file):
    """Extract text from a single PDF file-like object."""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def read_docx(docx_file):
    """Extract text from a single DOCX file-like object."""
    doc = Document(docx_file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)



def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.25
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer based on context.\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "If unknown, say 'Answer not available'."
        ),
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)


def user_input(user_question):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists("faiss_index"):
        st.error("No data found. Please upload and process files first.")
        return

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    docs = new_db.max_marginal_relevance_search(
        query=user_question,
        k=8,
        fetch_k=12,
        lambda_mult=0.5,
    )

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )
    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config(page_title="Chat with PDF/DOCX")
    st.header("📑 Chat with PDF & DOCX")

    with st.sidebar:
        st.title("Menu")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files and click Submit",
            accept_multiple_files=True,
            type=["pdf", "docx"],
        )
        if st.button("Submit and Process") and uploaded_files:
            with st.spinner("Processing..."):
                raw_text = get_file_text(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete!")

    user_question = st.text_input("Ask a question from the uploaded files")
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
