import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBeCE6hJjyQSzGUyCtlbpX2SLVeckAYGuE"

# Streamlit Setup
st.set_page_config(page_title="Google Review Chatbot")
st.title("🤖 Google Employee Review Chatbot")

@st.cache_resource
def load_chatbot():
    st.write("📦 Loading embedding model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    st.write("📦 Loading vector store from disk...")
    db = Chroma(persist_directory="chroma_local", embedding_function=embedding)

    st.write("🌐 Loading Gemini LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    st.write("🧠 Preparing QA chain...")
    prompt = PromptTemplate(
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
        chain_type_kwargs={"prompt": prompt}
    )

    return qa

try:
    qa_chain = load_chatbot()
    st.success("✅ Chatbot is ready.")
except Exception as e:
    st.error(f"Setup failed: {e}")

# Input Interface
query = st.text_input("Ask a question about working at Google:")
if query:
    with st.spinner("Thinking..."):
        try:
            answer = qa_chain.run(query)
            st.markdown("### Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Query failed: {e}")
