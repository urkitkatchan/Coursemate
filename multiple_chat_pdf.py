import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

# Configure the Google API
google_api = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api)

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to build a conversational chain
def get_conversational_chain():
    prompt_template = """You are a highly knowledgeable and empathetic dietitian assistant. Your role is to provide personalized dietary advice based on user input and the context derived from uploaded documents. 

- Respond with detailed, actionable dietary suggestions when the context is available.
- If the information is not in the provided context, politely state, \"The information is not available in the context provided.\"
- Keep responses user-friendly, scientifically accurate, and considerate of individual dietary needs such as allergies, dietary restrictions, and health goals.
- Avoid providing medical advice and always encourage users to consult a certified dietitian or healthcare professional for specific health concerns.

    Context:\n{context}\n
    Question:\n{question}\n
    """

    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)

    return chain

# Function to process user input question
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response['output_text']

# Function to generate a downloadable text file
def generate_text_file(content):
    return content

# Function to generate a downloadable PDF file
def generate_pdf_file(content):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    return pdf.output(dest="S").encode("latin1")

# Main function to handle Streamlit app flow
def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Profistant_An AI assistant 🤖")

    # User question input
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Process and respond if a question is provided
    if user_question:
        response = user_input(user_question)
        st.write("Reply: ", response)

        # Download response as text or PDF
        st.download_button(
            label="Download Response as Text",
            data=generate_text_file(response),
            file_name="response.txt",
            mime="text/plain"
        )

        st.download_button(
            label="Download Response as PDF",
            data=generate_pdf_file(response),
            file_name="response.pdf",
            mime="application/pdf"
        )

    # Sidebar for file upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit", type=["pdf"], accept_multiple_files=True)

        # Process PDF files when the button is clicked
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
