from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
#temperature is a parameter that controls the randomness of the output. Higher values (e.g., 1.5) will make the output more random, while lower values (e.g., 0.2) will make it more focused and deterministic.
#max_completion_tokens is a parameter that limits the number of tokens in the generated response. Setting it to a low value (e.g., 10) will result in shorter responses, while a higher value (e.g., 100) will allow for more detailed answers.
result = model.invoke("Define LCM meta research")

print(result.content)
print(result.metadata)
print(result.tokens)
print(result.usage)
print(result.cost)
print(result.model)

