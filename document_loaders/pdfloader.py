from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document_loaders/GRU.pdf") 
docss = data.load()

spiltter =  TokenTextSplitter(
    chunk_size =100,
    chunk_overlap = 10
)

chunk = spiltter.split_documents(docss)


print((chunk[0]).page_content)