import os
os.environ['HF_HOME']=r'F:\Langchain\models'

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    "My male cat's name is Leo and he is my best friend. He has thick orange fur, sharp golden eyes, and a confident personality that makes him look like a little lion.",

    "Leo loves sitting on the balcony in the early morning sunlight. He stretches his body lazily and watches the world outside as if he owns the entire neighborhood.",

    "Every morning, my cat Leo wakes me up by meowing softly near my pillow. If I ignore him, he gently taps my hand until I finally get out of bed.",

    "Leo enjoys eating grilled chicken and dry cat food. Whenever he hears the food packet opening, he runs quickly into the kitchen with excitement.",

    "In the evening, my male cat prefers to sit beside me while I work on my laptop. Sometimes he tries to step on the keyboard as if he wants attention.",

    "Leo is very protective and alert. If he hears a strange noise outside the door, he immediately stands still and listens carefully before investigating.",

    "My cat loves playing with a blue string toy and often jumps high into the air to catch it. His energy level is surprisingly high for such a relaxed-looking cat.",

    "During rainy days, Leo sleeps peacefully on the sofa, curling his tail around his body. He looks extremely calm and comfortable in his favorite spot.",

    "Leo has been part of my life for four years. He is not just a pet but a loyal companion who understands my emotions and stays close to me whenever I feel stressed."
]

query=input("enter a query: " )
query_vectors=embeddings.embed_query(query)
docs_vectors=embeddings.embed_documents(docs)

similarities = cosine_similarity([query_vectors], docs_vectors)[0]
best_index = np.argmax(similarities)


print(best_index)
print(similarities[best_index])
print(docs[best_index])

