import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
import google.generativeai as genai
from savechat import save_chat_history


load_dotenv()

# Configure the Google API
google_api = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api)

# Function to load or create FAISS index
def load_or_create_vector_store(text_chunks=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    elif text_chunks:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    else:
        raise ValueError("No FAISS index found and no text chunks provided.")
    return vector_store

# Function to build a conversational chain
def get_conversational_chain():
    prompt_template = """You are a highly knowledgeable and empathetic course assistant. Your role is to provide detailed answers to students based on the provided context. 
    Ensure your answers are clear, well-structured, and tailored to the user's query. Include additional relevant information to make the response comprehensive.

        If the answer is not available in the provided context, politely state, \"Answer is not available in the context provided.\"

    Context:\n{context}\n
    Question:\n{question}\n
    Response:\n"""
    
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)

    return chain

# Function to process user input question
def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response['output_text']

# Main function to handle Streamlit app flow
def main():
    st.set_page_config(page_title="Course mate", layout="centered")
    st.title("Coursemate(🧑🏻‍💻👩🏻‍💻)")

    # Add custom styles
    st.markdown("""
<style>
    /* Set the background for the main container */
    .stApp {
        background-color: #000000; /* Black */
        color: #FFFFFF; /* Dark blue for text */
    }

    /* Optional: Change sidebar background */
    .sidebar .sidebar-content {
        background-color: #E6E6FA; /* Lavender */
    }
</style>
""", unsafe_allow_html=True)


    with st.sidebar:
        st.header("Knowledge Base Options")
        use_existing_index = st.checkbox("Use Existing Knowledge Base", value=True)
        if not use_existing_index:
            st.warning("This version of the assistant only supports loading an existing FAISS index.")

    # Load FAISS index
    if use_existing_index:
        with st.spinner("Loading existing knowledge base..."):
            vector_store = load_or_create_vector_store()
            st.sidebar.success("Knowledge base loaded successfully!")

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question related to CS576 subject:", key="input_box")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Enter") and user_question:
            with st.spinner("Fetching response..."):
                response = user_input(user_question, vector_store)
                st.session_state.chat_history.append({"user": user_question, "bot": response})
                save_chat_history(st.session_state.chat_history)

    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []

    # Display chat history
    st.markdown("### Chat History")
    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Assistant:** {chat['bot']}")

if __name__ == "__main__":
    main()
