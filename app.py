import streamlit as st
st.set_page_config(page_title="Healthcare Chatbot")  # 🔥 MUST be first Streamlit command

from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Safe downloads (check if already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the lightweight GPT-2 model (runs on CPU)
@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="sshleifer/tiny-gpt2")

chatbot = load_chatbot()

# Define healthcare-specific chatbot logic
def healthcare_chatbot(user_input):
    user_input_lower = user_input.lower()

    if "symptom" in user_input_lower:
        return "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in user_input_lower:
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input_lower:
        return "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor."
    else:
        # Generate a response using GPT-2
        response = chatbot(user_input, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)
        return response[0]['generated_text'].strip()

# Streamlit UI
def main():
    st.title("🩺 Healthcare Assistant Chatbot")

    user_input = st.text_input("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input.strip():
            st.markdown(f"**You:** {user_input}")
            response = healthcare_chatbot(user_input)
            st.markdown(f"**Healthcare Assistant:** {response}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
