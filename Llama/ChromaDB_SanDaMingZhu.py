# https://blog.csdn.net/engchina/article/details/131868860

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

loader = TextLoader("A:/LangChain/Llama/docs/sandamingzhu.txt", encoding="utf-8")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="shibing624/text2vec-base-chinese")
# embedding_function = SentenceTransformerEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
# embedding_function = SentenceTransformerEmbeddings(model_name="moka-ai/m3e-large")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

# query it
# query = "白骨精被打死几次？"
# docs = db.similarity_search(query, k=3) # default k is 4

# print(len(docs))

# # print results
# for doc in docs:
#     print("="*100)
#     print(doc.page_content)

# query it
query = "刘关张在桃园做什么？"
docs = db.similarity_search(query, k=3) # default k is 4

print(len(docs))

# print results
for doc in docs:
    print("="*100)
    print(doc.page_content)

# save to disk
db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
docs = db2.similarity_search(query, k=1) # default k is 4

print(docs[0].page_content)


# load from disk
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db3.similarity_search(query, k=1) # default k is 4

print(docs[0].page_content)


# 4. 将 Chroma Client 传递到 Langchain ​
# 您还可以创建一个Chroma Client并将其传递给LangChain。如果您希望更轻松地访问底层数据库，这尤其有用。

# 您还可以指定您希望 LangChain 使用的集合名称。
# import chromadb

# persistent_client = chromadb.PersistentClient()
# collection = persistent_client.get_or_create_collection("collection_name")
# collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

# langchain_chroma = Chroma(
#     client=persistent_client,
#     collection_name="collection_name",
#     embedding_function=embedding_function,
# )

# print("There are", langchain_chroma._collection.count(), "in the collection")


# # create simple ids
# ids = [str(i) for i in range(1, len(docs) + 1)]

# # add data
# example_db = Chroma.from_documents(docs, embedding_function, ids=ids)
# # print(example_db)
# docs = example_db.similarity_search(query)
# print(docs[0].metadata)

# update the metadata for a document
# docs[0].metadata = {
#     "source": "./sidamingzhu.txt",
#     "new_value": "你好，世界！",
# }
# example_db.update_document(ids[0], docs[0])
# print(example_db._collection.get(ids=[ids[0]]))
# # delete the last document
# print("count before", example_db._collection.count())
# example_db._collection.delete(ids=[ids[-1]])
# print("count after", example_db._collection.count())
