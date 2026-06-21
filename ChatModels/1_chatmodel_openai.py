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

model2 = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5, max_completion_tokens=20)
result2 = model2.invoke("What is the capital of France?")   
print(result2.content)
print(result2.metadata)
print(result2.tokens)
print(result2.usage)
print(result2.cost)
print(result2.model)
print(model2.model)
print(model2.temperature)
print(model2.max_completion_tokens)
print(model2.invoke("What is the capital of France?").content)
def test_chat_model():
    model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
    result = model.invoke("Define LCM meta research")
    assert result.content is not None
    assert result.metadata is not None
    assert result.tokens is not None
    assert result.usage is not None
    assert result.cost is not None
    assert result.model == 'gpt-4'
    model2 = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5, max_completion_tokens=20)
    result2 = model2.invoke("What is the capital of France?")
    assert result2.content is not None
    assert result2.metadata is not None
    assert result2.tokens is not None
    assert result2.usage is not None
    assert result2.cost is not None
    assert result2.model == 'gpt-3.5-turbo'
    assert model2.model == 'gpt-3.5-turbo'
    assert model2.temperature == 0.5
    assert model2.max_completion_tokens == 20
    assert model2.invoke("What is the capital of France?").content == "The capital of France is Paris."

def test_chat_model_invalid_model():
    try:
        model = ChatOpenAI(model='invalid-model', temperature=1.5, max_completion_tokens=10)
        model.invoke("Define LCM meta research")
    except Exception as e:
        assert str(e) == "Model 'invalid-model' is not supported."

def test_chat_model_invalid_temperature():
    try:
        model = ChatOpenAI(model='gpt-4', temperature=-1.0, max_completion_tokens=10)
        model.invoke("Define LCM meta research")
    except Exception as e:
        assert str(e) == "Temperature must be between 0 and 2."
def test_chat_model_invalid_max_completion_tokens():
    try:
        model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=-10)
        model.invoke("Define LCM meta research")
    except Exception as e:
        assert str(e) == "Max completion tokens must be a positive integer."

q