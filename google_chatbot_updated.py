import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load and filter for Google
df = pd.read_csv("employee_reviews.csv", encoding='ISO-8859-1')
df = df[df['company'].str.lower() == 'google'].copy()

# Combine and format reviews better
def format_review(row):
    return f"""Rating: {row['overall-ratings']}
Pros: {row['pros']}
Cons: {row['cons']}
"""

df['text'] = df.apply(format_review, axis=1)

# Convert to LangChain documents
documents = [Document(page_content=text) for text in df['text'].tolist()]

# ✅ Use better chunker
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Store to ChromaDB (delete chroma_local/ first if re-running)
vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="chroma_local")
print("✅ Stored", len(split_docs), "chunks.")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")  # ✅ Stronger than small
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=200)
llm = HuggingFacePipeline(pipeline=pipe)

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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

# Reload vectorstore
db = Chroma(persist_directory="chroma_local", embedding_function=embedding)

# ✅ Retrieve more chunks (k=5 instead of 2)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

print("\n🤖 Google Employee Review Chatbot")
print("Type 'exit' to quit\n")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    answer = qa.run(query)
    print("🤖", answer)

