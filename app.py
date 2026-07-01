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

# Main Heading
st.title("📚 Turf Booking Knowledge Assistant")

st.write("Upload a Turf Booking PDF and ask questions about its contents.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload Turf Booking PDF",
    type=["pdf"]
)

# Initialize embeddings variable
embeddings = None

if uploaded_file is not None:
    # Create uploads folder if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    # Save uploaded PDF
    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File uploaded successfully: {uploaded_file.name}")

    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Display total pages
    st.write(f"Total Pages: {len(documents)}")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Display total chunks
    st.write(f"Total Chunks: {len(chunks)}")

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.success("Embedding model loaded successfully.")

    # Save chunks into ChromaDB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    st.success("Documents saved to ChromaDB successfully.")

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

        if not results:
            st.error("Information not found.")
        else:
            best_doc, score = results[0]

    
            

            # Check whether the answer is relevant
            if score < 0.10:
                st.error("Information not found.")
            else:
                st.subheader("Answer")
                st.write(best_doc.page_content)