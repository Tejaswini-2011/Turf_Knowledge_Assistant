import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Page Title
st.set_page_config(
    page_title="Turf Booking Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------- Company Logo ----------------
st.image("tachyon_logo.jpeg", width=180)

# ---------------- Sidebar ----------------
st.sidebar.title("📚 About Project")

st.sidebar.markdown("""
### Turf Booking Knowledge Assistant

This application allows users to:

- 📄 Upload a Turf Booking PDF
- ✂️ Split the document into chunks
- 🧠 Generate embeddings
- 💾 Store embeddings in ChromaDB
- ❓ Ask questions about the document
- 🔍 Retrieve relevant information using semantic search

---
### 🛠 Technologies Used

- Streamlit
- LangChain
- PyPDFLoader
- Hugging Face Embeddings
- ChromaDB

---
### 📌 Instructions

1. Upload a Turf Booking PDF.
2. Wait until processing is complete.
3. Enter your question.
4. Click **Ask** to retrieve information.
""")

# ---------------- Main Page ----------------

st.title("📚 Turf Booking Knowledge Assistant")

st.markdown("""
### AI-Powered PDF Question Answering System

Upload a Turf Booking PDF, process it using **RAG (Retrieval-Augmented Generation)**, and ask questions to retrieve relevant information from the document.
""")

st.divider()

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload Turf Booking PDF",
    type=["pdf"]
)

# Initialize embeddings variable
embeddings = None

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if uploaded_file is not None:

    with st.spinner("📄 Processing PDF... Please wait..."):

        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join("uploads", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # PDF Information
        file_name = uploaded_file.name
        file_size = round(uploaded_file.size / 1024, 2)
        total_pages = len(documents)

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = text_splitter.split_documents(documents)

        # Create embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Save chunks into ChromaDB
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

    # Display information after processing
    st.success(f"✅ File uploaded successfully: {file_name}")

    st.write(f"📄 File Name: {file_name}")
    st.write(f"📏 File Size: {file_size} KB")
    st.write(f"📃 Total Pages: {total_pages}")
    st.write(f"📦 Total Chunks: {len(chunks)}")

    st.success("✅ Embedding model loaded successfully.")
    st.success("✅ Documents saved to ChromaDB successfully.")

# Question Box
question = st.text_input(
    "Ask a question about the document"
)

# Ask Button
if st.button("Ask"):

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    elif question.strip() == "":
        st.warning("Please enter a question.")

    else:
        # Load existing ChromaDB
        vector_db = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        # Retrieve document with relevance score
        results = vector_db.similarity_search_with_relevance_scores(
            question,
            k=1
        )

        # Check if any result is returned
        if not results:
            st.error("❌ Information Not Found")

            st.info("""
We couldn't find a relevant answer in the uploaded PDF.

Try the following:
• Ask your question using different words.
• Use keywords available in the PDF.
• Make sure the uploaded PDF contains the required information.
""")

        else:
            best_doc, score = results[0]

            # Check whether the answer is relevant
            if score < 0.10:
                st.error("❌ Information Not Found")

                st.info("""
We couldn't find a relevant answer in the uploaded PDF.

Try the following:
• Ask your question using different words.
• Use keywords available in the PDF.
• Make sure the uploaded PDF contains the required information.
""")

            else:
                # Better Answer Card
                with st.container(border=True):
                    st.subheader("🤖 AI Answer")
                    st.markdown(best_doc.page_content)

                    # Display Source Page Number
                    page_number = best_doc.metadata.get("page", 0) + 1
                    st.caption(f"📄 Source Page: {page_number}")

                # Save Chat History
                st.session_state["chat_history"].append({
                    "question": question,
                    "answer": best_doc.page_content,
                    "page": page_number
                })

        # Clear Chat Button
if st.button("🗑️ Clear Chat"):
    st.session_state["chat_history"] = []
    st.success("Chat history cleared successfully!")

        # ---------------- Chat History ----------------

if st.session_state["chat_history"]:

    st.divider()
    st.subheader("💬 Chat History")

    for i, chat in enumerate(st.session_state["chat_history"], start=1):

        with st.expander(f"Question {i}: {chat['question']}"):

            st.write(chat["answer"])
            st.caption(f"📄 Source Page: {chat['page']}")


    st.divider()

st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>

Developed by <b>Sai Tejaswini</b><br><br>

📚 Turf Booking Knowledge Assistant<br>

Built with ❤️ using Streamlit • LangChain • Hugging Face • ChromaDB

</div>
""", unsafe_allow_html=True)