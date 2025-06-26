import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Read & Filter
df = pd.read_csv("employee_reviews.csv", encoding='ISO-8859-1')
df = df[df['company'].str.lower() == 'google'].copy()

# Step 2: Format
def format_review(row):
    return f"""Rating: {row['overall-ratings']}\nPros: {row['pros']}\nCons: {row['cons']}"""
df['text'] = df.apply(format_review, axis=1)

# Step 3: Convert to LangChain documents
documents = [Document(page_content=t) for t in df['text'].tolist()]

# Step 4: Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

# Step 5: Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Step 6: Save vectorstore
vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="chroma_local")
vectorstore.persist()

print("✅ Vectorstore saved to disk.")
