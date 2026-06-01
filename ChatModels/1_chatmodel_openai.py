from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
#temperature is a parameter that controls the randomness of the output. Higher values (e.g., 1.5) will make the output more random, while lower values (e.g., 0.2) will make it more focused and deterministic.

result = model.invoke("Define LCM meta research")

print(result.content)