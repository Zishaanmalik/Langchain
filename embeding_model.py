import os
os.environ['HF_HOME']=r'F:\Langchain\models'

from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text ="what is the capital of india"

docs=["delhi is the capital of india",
      "i love india",
      "i also want to visit africa"]

vectors=embeddings.embed_query(text)

doc_vectors = embeddings.embed_documents(docs)

print(str(vectors))

print(str(doc_vectors))