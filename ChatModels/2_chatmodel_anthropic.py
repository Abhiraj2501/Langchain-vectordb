from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

result = model.invoke('What is the capital of India')


print(result.content)

def test_chat_model_anthropic():
    model = ChatAnthropic(model='claude-3-5-sonnet-20241022')
    result = model.invoke('What is the capital of India')
    assert result.content is not None
    assert result.metadata is not None
    assert result.tokens is not None
    assert result.usage is not None
    assert result.cost is not None
    assert result.model == 'claude-3-5-sonnet-20241022'
test_chat_model_anthropic()

def test_chat_model_anthropic_invalid_model():
    try:
        model = ChatAnthropic(model='invalid-model')
        model.invoke('What is the capital of India')
    except Exception as e:
        assert str(e) == "Model 'invalid-model' is not supported."

test_chat_model_anthropic_invalid_model()
