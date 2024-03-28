import torch, sys, os
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts       import PromptTemplate
from langchain.chains        import LLMChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai                    import OpenAI, OpenAIEmbeddings

from langchain_community.llms       import LlamaCpp, CTransformers
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, WebBaseLoader, PyPDFDirectoryLoader, CSVLoader

openai_api_key= os.environ.get("OPENAI_API_KEY")
GPT_MODEL = "gpt-3.5-turbo-instruct"
EB_MODEL = "text-embedding-3-small"

def query_pdf(query):
    # Load document using PyPDFLoader document loader
    loader = PyPDFLoader("data/pdf/Python Programming - An Introduction To Computer Science.pdf")
    documents = loader.load()
    # Split document in chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # Create vectors
    vectorstore = FAISS.from_documents(docs, embeddings)
    # Persist the vectors locally on disk
    # vectorstore.save_local("faiss_index_constitution")

    # Load from local storage
    # persisted_vectorstore = FAISS.load_local("faiss_index_constitution", embeddings)

    # Use RetrievalQA chain for orchestration
    # qa = RetrievalQA.from_chain_type(llm=OpenAI(model=GPT_MODEL), chain_type="stuff", retriever=persisted_vectorstore.as_retriever())
    qa = RetrievalQA.from_chain_type(llm=OpenAI(model=GPT_MODEL), chain_type="stuff", retriever=vectorstore.as_retriever())
    result = qa.run(query)
    print(result)


def main():
    query = input("Type in your query: \n")
    while query != "exit":
        query_pdf(query)
        query = input("Type in your query: \n")


if __name__ == "__main__":
    main() 