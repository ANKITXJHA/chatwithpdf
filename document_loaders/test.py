from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 10,
    chunk_overlap= 1
)



data = TextLoader("document_loaders/notes.txt")
docs =data.load()


chunk = splitter.split_documents(docs)
print(len(chunk))