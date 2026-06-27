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