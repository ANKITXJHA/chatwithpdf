from langchain_community.document_loaders import WebBaseLoader
url = "https://www.news18.com/world/flydubai-omani-co-pilot-identified-fresh-details-emerge-on-syria-links-extremist-views-ws-l-10365224.ht"


data = WebBaseLoader(url)
docs = data.load()
print(docs[0].page_content) 