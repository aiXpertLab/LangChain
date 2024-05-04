import sys
from langchain_community.llms       import LlamaCpp, CTransformers
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader, PyPDFDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

# load PDF files from a directory
loader = PyPDFDirectoryLoader("data/")
data = loader.load()
# print the loaded data, which is a list of tuples (file name, text extracted from the PDF)
print(data)

# split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
# print the number of chunks obtained
len(text_chunks)

# download the embeddings to use to represent text chunks in a vector space, using the pre-trained model "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# create embeddings for each text chunk using the FAISS class, which creates a vector index using FAISS and allows efficient searches between vectors
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

# Import the neural language model using the LlamaCpp class, which allows you to use a GPT-3 model in C++ with various parameters such as temperature, top_p, verbose and n_ctx (maximum number of tokens that can be generated)
llm = LlamaCpp(
    streaming = True,
    model_path = "/mnt/e/models/llama/llama-2-7b-chat.Q4_0.gguf",
    temperature=0.75,
    top_p=1,
    verbose=True,
    n_ctx=4096
)

# Create a question answering system based on information retrieval using the RetrievalQA class, which takes as input a neural language model, a chain type and a retriever (an object that allows you to retrieve the most relevant chunks of text for a query)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}))

# define a query to ask the system
query = "What is linear regression model"
# run the system and get a response
qa.run(query)

while True:
  user_input = input(f"Input Prompt: ")
  if user_input == 'exit':
    print('Exiting')
    sys.exit()
  if user_input == '':
    continue
  # pass the query to the system and print the response
  result = qa({'query': user_input})
  print(f"Answer: {result['result']}")