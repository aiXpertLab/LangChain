from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores     import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_community.embeddings    import ModelScopeEmbeddings

from langchain.text_splitter import CharacterTextSplitter
import chardet

loader1 = WebBaseLoader("https://aixpertlab.github.io/data/AnnaKarenina1.txt")
loader2 = WebBaseLoader("https://aixpertlab.github.io/data/Wuthering_Heights.txt")
data1 = loader1.load()
data2 = loader2.load()

text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=0)
documents_sanguo = text_splitter.split_documents(data1)
documents_xiyou = text_splitter.split_documents(data2)
documents = documents_sanguo + documents_xiyou
print("documents nums:", documents.__len__())

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
db = Chroma.from_documents(documents, embedding=embeddings)

query = "Russian"
docs = db.similarity_search(query, k=3)

for doc in docs:
    print("===")
    print("metadata:", doc.metadata)
    print("page_content:", doc.page_content)