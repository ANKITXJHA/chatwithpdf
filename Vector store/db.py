from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

from langchain_core.documents import Document

docs = [
    Document(page_content="Artificial Intelligence is the field of building machines that can perform tasks that normally need human intelligence.", metadata={"topic": "AI"}),
    Document(page_content="Machine Learning is a part of AI where models learn patterns from data instead of being explicitly programmed.", metadata={"topic": "AI"}),
    Document(page_content="Python is a popular programming language for AI because of its simple syntax and libraries like NumPy, Pandas and TensorFlow.", metadata={"topic": "Python"}),
]




embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embeddings_model,
    persist_directory="chromaDB"

)

result =  vectorstore.similarity_search("What is used for data analysis?",k=2)
for r in result:
    print(r.page_content)
    print(r.metadata)

retevier = vectorstore.as_retriever()

docs = retevier.invoke("explain Machine learning")

for d in docs:
    print(d.page_content)
  