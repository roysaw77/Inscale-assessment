import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch

# CONFIG


CHROMA_PATH = "./chroma_db"
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

documents = []

# LOAD FILES
# Merged Files
segments_df = pd.read_csv("agora/segments.csv")
documents_df = pd.read_csv("agora/documents.csv")

documents_df = documents_df[
    [
        "AGORA ID",
        "Official name",
        "Authority",
        "Collections",
        "Short summary"
    ]
]
merged_df = segments_df.merge(
    documents_df,
    left_on="Document ID",
    right_on="AGORA ID",
    how="left"
)



documents = []

for col, row in merged_df.iterrows():

    text = f"""
            Title: {row['Official name']}

            Authority: {row['Authority']}

            Collection: {row['Collections']}

            Summary:
            {row['Short summary']}

            Content:
            {row['Text']}
            """

    if not text.strip():
        continue

    documents.append(
        Document(
            page_content=text,
        )
    )

print(f"Loaded {len(documents)} segments")

# EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={
        "device": "cuda"
    }
)

# VECTOR STORE

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)

print("Vector database created!")