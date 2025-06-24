import os
import pandas as pd
import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔐 API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBeCE6hJjyQSzGUyCtlbpX2SLVeckAYGuE"  # 🔁 Replace with your actual key

# ✅ Streamlit title loads FIRST
st.set_page_config(page_title="Google Review Chatbot", page_icon="🤖")
st.title("🤖 Google Employee Review Chatbot")
st.write("✅ Loading chatbot components...")

# ✅ Heavy setup cached
@st.cache_resource
def load_qa_chain():
    st.write("📄 Reading CSV...")
    df = pd.read_csv("employee_reviews.csv", encoding='ISO-8859-1')
    
    st.write("🔍 Filtering for 'Google'...")
    df = df[df['company'].str.lower() == 'google'].copy()
    st.write(f"✅ {len(df)} reviews found")

    st.write("🧹 Formatting reviews...")
    def format_review(row):
        return f"""Rating: {row['overall-ratings']}\nPros: {row['pros']}\nCons: {row['cons']}"""
    df['text'] = df.apply(format_review, axis=1)

    st.write("📘 Creating documents...")
    documents = [Document(page_content=t) for t in df['text'].tolist()]

    st.write("✂️ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    st.write(f"🧩 {len(split_docs)} chunks created")

    st.write("🔍 Loading embeddings model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    st.write("🧠 Building Chroma vectorstore...")
    vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="chroma_local")

    st.write("🔁 Reloading Chroma from disk...")
    db = Chroma(persist_directory="chroma_local", embedding_function=embedding)

    st.write("🌐 Loading Gemini LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    st.write("📄 Preparing prompt and QA chain...")
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant analyzing Google employee reviews.

Use the following review excerpts to answer the user's question.
Only use the information provided — do not assume anything.

--------------------
{context}
--------------------

Question: {question}

Answer:"""
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )

    st.success("✅ QA chain loaded.")
    return qa
# 🔁 Load QA chain (Streamlit caches this)
try:
    qa_chain = load_qa_chain()
    st.success("Chatbot is ready.")
except Exception as e:
    st.error(f"Setup failed: {e}")

# ✅ Input UI
query = st.text_input("Ask a question about working at Google:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = qa_chain.run(query)
            st.markdown("### Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Query failed: {e}")
