import hashlib
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF (RAG)")


# Load the embedding model only once (it is slow to load)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."

Context:
{context}
""",
        ),
        ("human", "{question}"),
    ]
)


def build_vectorstore(uploaded_files):
    """Read all uploaded PDFs, split them into chunks and store them in an in-memory Chroma DB."""
    all_docs = []
    for uploaded in uploaded_files:
        # PyPDFLoader needs a file path, so save the upload to a temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        pages = PyPDFLoader(tmp_path).load()
        for page in pages:
            # Show the real file name in the sources instead of the temp path
            page.metadata["source"] = uploaded.name
        all_docs.extend(pages)
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    if not chunks or not any(c.page_content.strip() for c in chunks):
        return None, 0

    # No persist_directory: the DB lives in memory and is rebuilt for each new upload.
    # A unique collection name avoids mixing chunks from an old PDF with a new one.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"pdf_{uuid.uuid4().hex}",
    )
    return vectorstore, len(chunks)


# ---------------- Sidebar: upload ----------------
with st.sidebar:
    st.header("Upload PDF")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files", type="pdf", accept_multiple_files=True
    )
    st.caption("Answers will come only from the PDFs you upload here.")

    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()

if not uploaded_files:
    st.info("Please upload a PDF from the sidebar, then ask your question.")
    st.stop()

# Rebuild the vector DB only when the set of uploaded files changes
files_key = hashlib.md5(
    "".join(f"{f.name}{f.size}" for f in uploaded_files).encode()
).hexdigest()

if st.session_state.get("files_key") != files_key:
    with st.spinner("Processing your PDF (chunking and creating embeddings)..."):
        vectorstore, n_chunks = build_vectorstore(uploaded_files)

    if vectorstore is None:
        st.error(
            "Could not extract any text from this PDF. It may be a scanned or image-based PDF (OCR is required)."
        )
        st.stop()

    st.session_state.vectorstore = vectorstore
    st.session_state.files_key = files_key
    st.session_state.messages = []  # new PDF means a fresh chat
    st.success(f"Ready! Created {n_chunks} chunk{'s' if n_chunks != 1 else ''} from your PDF.")

retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
)

# ---------------- Chat history ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(s)

# ---------------- New question ----------------
query = st.chat_input("Ask a question about your PDF...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            final_prompt = prompt.invoke({"context": context, "question": query})
            response = get_llm().invoke(final_prompt)

        st.markdown(response.content)

        # Build source labels like "file.pdf - page 12"
        sources = []
        seen = set()
        for doc in docs:
            label = f"**{doc.metadata.get('source', 'PDF')}** - page {doc.metadata.get('page', 0) + 1}"
            if label not in seen:
                seen.add(label)
                sources.append(label)

        with st.expander("Sources"):
            for s in sources:
                st.markdown(s)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content, "sources": sources}
    )