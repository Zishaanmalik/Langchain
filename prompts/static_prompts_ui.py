import os
os.environ['HF_HOME']=r'F:\Langchain\MODELS'

from langchain_huggingface import  ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2-1.5B-Instruct",
    task="text-generation",
    #max_new_tokens=100
)
model =ChatHuggingFace(llm=llm)

st.header('Reasearch Tool')
user_input=st.text_input('Enter your prompt')

if st.button('summrize'):
    result=model.invoke(user_input)
    st.write(result.content)



