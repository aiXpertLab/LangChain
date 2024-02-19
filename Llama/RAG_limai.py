# https://zhuanlan.zhihu.com/p/668082024

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

loader = TextLoader("A:/LangChain/Llama/docs/limai.txt")
documents = loader.load()
print(len(documents))
text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=0)
documents = text_splitter.split_documents(documents)
print(len(documents))

# embedding model: m3e-base
model_name = "moka-ai/m3e-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="为文本生成向量表示用于文本检索"
            )

# load data to Chroma db
db = Chroma.from_documents(documents, embedding)
# similarity search
db.similarity_search("藜一般在几月播种？")