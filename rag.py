from openai import OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from streamlit import context

load_dotenv()

def load_vector_db():

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )

    return db

def retrieve_context(db, query, k=5):
    results = db.similarity_search(
        query,
        k=k
    )
    context = "\n\n".join(
        [result.page_content for result in results]
    )
    return context, results

def generate_answer(context, query):

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    prompt = f"""
    You are an AI assistant.

    Answer using the provided context.
    If the answer is not contained within the context, say you don't know.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    completion = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content

