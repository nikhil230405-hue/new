import streamlit as st
import faiss
import numpy as np
import pickle
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ------------------------
# Load ML classification model
# ------------------------
model = pickle.load(open("question_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def classify_question(q):
    vector = tfidf.transform([q])
    return model.predict(vector)[0]

# ------------------------
# LLM Client
# ------------------------
client = Groq(api_key="YOUR_GROQ_API_KEY")

# ------------------------
# Streamlit UI
# ------------------------
st.title("📘 University FAQ RAG Chatbot (Groq + FAISS + ML Classification)")

uploaded_file = st.file_uploader("Upload your FAQ PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully!")

    # Load PDF text
    loader = PyPDFLoader(uploaded_file)
    pages = loader.load()

    # Text chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    # Embedding model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunk_embeddings = embedder.encode([chunk.page_content for chunk in chunks])

    # FAISS Index
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(chunk_embeddings))

    # User question
    question = st.text_input("Ask a question:")

    if st.button("🟦 Answer"):
        if question.strip() != "":
            
            # ------------------------
            # Step 1: Question Classification
            # ------------------------
            q_type = classify_question(question)
            st.write(f"📌 **Detected Question Type:** `{q_type}`")

            # ------------------------
            # Step 2: Retrieve chunks (RAG)
            # ------------------------
            q_embed = embedder.encode([question])
            D, I = index.search(np.array(q_embed), 3)

            retrieved_chunks = [chunks[i].page_content for i in I[0]]

            combined_context = "\n\n".join(retrieved_chunks)

            # ------------------------
            # Step 3: LLM Answer
            # ------------------------
            prompt = f"""
            You are a helpful university FAQ assistant.
            Answer the user question using ONLY the context provided.

            Context:
            {combined_context}

            Question: {question}
            """

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            st.subheader("🟦 Answer")
            st.write(response.choices[0].message.content)

            # Show retrieved chunks
            st.subheader("🔍 Retrieved Chunks Used")
            for i, ch in enumerate(retrieved_chunks):
                st.text_area(f"Chunk {i+1}", ch, height=120)
