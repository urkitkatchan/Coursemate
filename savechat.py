import json
import os

# Function to save chat history to a file
def save_chat_history(chat_history, file_path="chat_history.json"):
    """Save the chat history to a JSON file.

    Args:
        chat_history (list): List of dictionaries containing the chat history.
        file_path (str): Path to the file where the chat history will be saved.

    Returns:
        None
    """
    try:
        with open(file_path, "a") as file:
            json.dump(chat_history, file, indent=4)
        print(f"Chat history saved successfully to {file_path}.")
    except Exception as e:
        print(f"Error saving chat history: {e}")
