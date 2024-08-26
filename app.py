import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to extract text from PDFs using PyMuPDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_bytes = pdf.read()  # Read the uploaded file as bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # Open the PDF from bytes

        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")  # Extract text with basic layout, including tables
            text += page_text + "\n"

    return text


# Function to split text into chunks with increased overlap
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1500)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create and save a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Function to set up the conversational chain
# Function to set up the conversational chain with an enhanced prompt template
def get_conversational_chain():
    prompt_template = """
    You are an AI assistant designed to help users analyze and understand documents. Answer the question as detailed as possible using the provided context. 
    Be flexible in your response style to match the user's input pattern. If the question is in layman terms, respond accordingly. If the question is technical, maintain the technical tone.
    If the answer is not in the provided context, simply say, "The answer is not available in the context." Do not provide incorrect or speculative answers.

    Context:
    {context}
    
    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and retrieve answers
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


# Main function for the Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.warning("Please upload PDF files first.")


if __name__ == "__main__":
    main()
