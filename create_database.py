from dotenv import load_dotenv
load_dotenv()
 
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# PDF DATA
pdfdata = PyPDFLoader("document_loaders/DeepLearningBook.pdf")
docspdf = pdfdata.load()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
   
)
chunks = text_splitter.split_documents(docspdf)

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings_model,
    persist_directory="chromaDB"

)