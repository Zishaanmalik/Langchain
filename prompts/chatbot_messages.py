import os
os.environ['HF_HOME']=r'F:\Langchain\models'
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import  ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    #max_new_tokens=100
)
model =ChatHuggingFace(llm=llm)
history=[
    SystemMessage(content="your are ai assistant answer all the questions asked")
]
while True:

    user_input=input('you: ')
    if user_input.lower() == 'exit':
        break
    history.append(HumanMessage(content=user_input))
    result=model.invoke(history)
    history.append(AIMessage(content=result.content))
    print('AI: ',result.content)



