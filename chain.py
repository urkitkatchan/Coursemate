
# chain.py
from docs import DocumentManager
from vectorstore import VectorStoreManager
from chatbot import DietitianChatbot
import os

class DietitianAssistant:
    def __init__(self, data_directory, store_path='vectorstore.faiss'):
        self.doc_manager = DocumentManager(data_directory)
        self.vector_store_manager = VectorStoreManager(store_path)
        self.chatbot = DietitianChatbot(self.vector_store_manager)

    def compile_knowledge_base(self):
        if not os.path.exists(self.vector_store_manager.store_path):
            self.vector_store_manager.compile_and_store_knowledge(self.doc_manager.directory)
        else:
            self.vector_store_manager.load_vector_store()

    def get_response(self, query):
        return self.chatbot.generate_response(query)
