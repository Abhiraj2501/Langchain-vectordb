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
check_string = random_string.replace("random", "sample")

accuracy = 0.95
def calculate_accuracy(predictions, ground_truth):
    correct_predictions = sum(p == gt for p, gt in zip(predictions, ground_truth))
    return correct_predictions / len(ground_truth) if ground_truth else 0.0

calculate_accuracy([1, 0, 1, 1], [1, 0, 0, 1])

second_string = "This is another string for testing."
check_second_string = second_string.replace("another", "different")
calculate_accuracy([1, 1, 0, 1], [1, 1, 1, 1])

new_accuracy = calculate_accuracy([1, 0, 1, 1], [1, 0, 0, 1])

