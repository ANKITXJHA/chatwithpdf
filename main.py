from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

emeddings_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory = "chromaDB",
    embedding_function = emeddings_model,
)

retrvier = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5,
    }
)

llm = ChatGroq(model  = "openai/gpt-oss-120b")

#PROMPT TEMMPLATE
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."

Context:{context}
"""),

    ("human", "{question}"),
    ]
)

print("RAG SYSTEM CREATED")

print("press 0 to exit")

while True:
    query = input("You :")
    if query =="0" :
     break
    docs = retrvier.invoke(query)


    context = "\n\n".join(
       [doc.page_content for doc in docs]

    )
    final_prompt = prompt.invoke({
       "context" : context,
       "question" : query,
    })
    response =  llm.invoke(final_prompt)
    print(f"\n AI: {response.content}")