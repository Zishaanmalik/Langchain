import os
os.environ['HF_HOME']=r'F:\Langchain\models'

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model =ChatHuggingFace(llm=llm)

result=model.invoke("what ist capital of india")

print(result.content)
print(result)

