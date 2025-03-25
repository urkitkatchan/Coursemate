import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
from savechat import save_chat_history

load_dotenv()

# Configure Google API
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
    prompt_template = """You are a highly knowledgeable and empathetic dietitian assistant. Your role is to provide detailed, actionable dietary advice based on the provided context. 
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
    st.set_page_config(page_title="Dietitian Assistant", layout="wide")
    st.title("Dietitian Assistant Chat")

    # Sidebar options for loading FAISS index
    with st.sidebar:
        st.header("Knowledge Base")
        use_existing_index = st.checkbox("Use Existing Knowledge Base", value=True)
        if not use_existing_index:
            st.warning("Currently, only an existing FAISS index can be loaded.")

    # Load FAISS index
    if use_existing_index:
        with st.spinner("Loading knowledge base..."):
            vector_store = load_or_create_vector_store()
            st.sidebar.success("Knowledge base loaded successfully!")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input section
    st.text_input("Ask a question related to diet and nutrition:", key="user_input")

    # Buttons for sending input or clearing chat
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Send") and st.session_state.user_input:
            with st.spinner("Generating response..."):
                response = user_input(st.session_state.user_input, vector_store)
                st.session_state.chat_history.append({"user": st.session_state.user_input, "bot": response})
                save_chat_history(st.session_state.chat_history)
                st.session_state.user_input = ""  # Clear input box

    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []

    # Display chat history
    st.markdown("### Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['bot']}")
        st.divider()

if __name__ == "__main__":
    main()
