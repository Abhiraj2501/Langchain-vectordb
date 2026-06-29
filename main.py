import langchain 
print(langchain.__version__)

def test_llm():
    llm = OpenAI(model="gpt-3.5-turbo-instruct")
    result = llm.invoke("what is the use of langchain?")
    assert result.content is not None
    assert result.metadata is not None
    assert result.tokens is not None
    assert result.usage is not None
    assert result.cost is not None
    assert result.model == "gpt-3.5-turbo-instruct"
def test_llm_with_custom_model():
    llm = OpenAI(model="gpt-4o")
    result = llm.invoke("what is the use of langchain?")
    assert result.content is not None
    assert result.metadata is not None
    assert result.tokens is not None
    assert result.usage is not None
    assert result.cost is not None
    assert result.model == "gpt-4o"

test_llm()
test_llm_with_custom_model()

random_string = "This is a random string for testing purposes."
new_string = random_string.replace("random", "sample")

test_llm()

random_string = "This is a random string for testing purposes."