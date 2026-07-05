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

r2 = 0.85
def calculate_r2(predictions, ground_truth):
    mean_gt = sum(ground_truth) / len(ground_truth) if ground_truth else 0.0
    ss_total = sum((gt - mean_gt) ** 2 for gt in ground_truth)
    ss_residual = sum((gt - p) ** 2 for p, gt in zip(predictions, ground_truth))
    return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
def test_r2():
    predictions = [3, -0.5, 2, 7]
    ground_truth = [2.5, 0.0, 2, 8]
    r2_value = calculate_r2(predictions, ground_truth)
    assert r2_value >= 0 and r2_value <= 1
    print(f"R² value: {r2_value}")

def test_r2_with_custom_data():
    predictions = [2, 3, 5, 7]
    ground_truth = [2, 3, 5, 7]
    r2_value = calculate_r2(predictions, ground_truth)
    assert r2_value == 1.0
    print(f"R² value with custom data: {r2_value}")

def test_r2_with_edge_case():
    predictions = [1, 1, 1, 1]
    ground_truth = [1, 1, 1, 1]
    r2_value = calculate_r2(predictions, ground_truth)
    assert r2_value == 1.0
    print(f"R² value with edge case: {r2_value}")