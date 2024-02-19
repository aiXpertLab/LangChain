import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(
    model_path="e:/models/llama/llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=40,
    n_batch=512,  # Batch size for model processing
    verbose=False,  # Enable detailed logging for debugging
)

# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

st.title("Chatbot Interface")

question = st.text_input("You:", "")

if st.button("Ask"):
    if question:
        answer = llm_chain.run(question)
        st.text_area("Bot:", value=answer, height=200)
    else:
        st.warning("Please enter a question.")
