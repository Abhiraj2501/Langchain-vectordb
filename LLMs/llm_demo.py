from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("what is the use of langchain?")

print(result)

def test_llm():
    llm = OpenAI(model="gpt-3.5-turbo-instruct")
    result = llm.invoke("what is the use of langchain?")
    assert result.content is not None
    assert result.metadata is not None
    assert result.tokens is not None
    assert result.usage is not None
    assert result.cost is not None
    assert result.model == "gpt-3.5-turbo-instruct"

test_llm()

def test_llm_with_custom_model():