# 📄 Chat with your PDF (RAG)

A Retrieval-Augmented Generation (RAG) app that lets you upload one or more PDFs and ask questions about them. Answers are generated **only from the uploaded documents**, and every answer shows the source file and page number.

Built with LangChain, Groq, Hugging Face embeddings, ChromaDB and Streamlit.

![App screenshot](screenshot.png)

## Features

- Upload one or multiple PDFs from the sidebar
- Answers are grounded in the uploaded documents only
- If the answer is not in the document, the app says so instead of guessing
- Source citations (file name and page number) with every answer
- MMR retrieval for diverse, less repetitive context
- The vector database is rebuilt automatically when new files are uploaded
- Embedding model and LLM are cached, so the app stays responsive

## How it works

```
PDF upload
   |
   v
PyPDFLoader  -->  RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
   |
   v
HuggingFace embeddings (all-MiniLM-L6-v2)  -->  Chroma (in-memory)
   |
   v
User question  -->  MMR retriever (k=4, fetch_k=10)  -->  Top relevant chunks
   |
   v
Prompt (context + question)  -->  Groq LLM (openai/gpt-oss-120b)  -->  Answer + sources
```

1. **Load:** each PDF is read page by page.
2. **Split:** pages are cut into overlapping chunks so context is not lost at chunk boundaries.
3. **Embed and store:** chunks are converted to vectors and stored in Chroma.
4. **Retrieve:** for each question, the 4 most relevant and diverse chunks are fetched using Maximal Marginal Relevance (MMR).
5. **Generate:** the chunks and the question are sent to the LLM with a prompt that restricts it to the provided context.

## Tech stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| PDF parsing | PyPDF |

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/ANKITXJHA/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add your Groq API key

Create a `.env` file in the project root (see `.env.example`):

```
GROQ_API_KEY=your_api_key_here
```

You can get a free API key from the [Groq Console](https://console.groq.com/).

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`), upload a PDF and start asking questions.

## Requirements

```
streamlit
python-dotenv
langchain-core
langchain-community
langchain-groq
langchain-huggingface
langchain-chroma
langchain-text-splitters
sentence-transformers
pypdf
```

## Limitations

- Scanned or image-only PDFs are not supported because there is no OCR step. The PDF must contain selectable text.
- Each question is answered independently. The app does not use previous chat messages as context for follow-up questions.
- The first run downloads the embedding model, so it takes longer to start.
- Very large PDFs take a few minutes to embed, and Groq's free tier has rate and token limits.

## Future improvements

- Conversation memory for follow-up questions
- OCR support for scanned PDFs
- Persistent vector storage so documents do not need to be re-processed
- Streaming responses
- Deployment on Streamlit Community Cloud or Hugging Face Spaces

## What I learned

- Why sending a whole document to an LLM fails (context and token limits) and how chunking plus retrieval solves it
- How MMR retrieval differs from plain similarity search
- Debugging real-world library issues (version mismatches, deprecated packages)
- Keeping secrets out of version control with `.env` and `.gitignore`

## Author

**Ankit Kumar Jha**
B.Tech CSE (AI & ML) student

- GitHub: [ANKITXJHA](https://github.com/ANKITXJHA)
- LinkedIn: [ankitjha21ya](https://linkedin.com/in/ankitjha21ya)
