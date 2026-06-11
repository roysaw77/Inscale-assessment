import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os


from rag import generate_answer, load_vector_db, retrieve_context

load_dotenv()

db = load_vector_db()

st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input(
    "Ask a question"
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    context, results = retrieve_context(
        db,
        query,
        k=3
    )

    answer = generate_answer(
        context,
        query
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    print(f"Context: {context}")