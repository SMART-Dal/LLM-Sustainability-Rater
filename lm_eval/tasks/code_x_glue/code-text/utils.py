import re
import datasets


def process_docs(dataset: datasets.Dataset):
    dataset = dataset.select([i for i in range(1000)])
    return dataset

def doc_to_text(doc):
    inputs = " ".join(doc["code_tokens"]).replace("\n", " ")
    inputs = " ".join(inputs.strip().split())
    prompt = """Generate a Python docstring for the following code. The docstring should follow best practices and clearly explain the code's purpose, its parameters (inputs), return value (output). Remember to only generate docstring, and DO NOT generate method or class header.\n\nMain code:\n\n```python\n{}```\n\nYour generated docstring:"""
    return prompt.format(inputs)
    

def clean_text(text):
    text.replace('\n', ' ').lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def doc_to_target(doc):
    # targets = " ".join(doc["docstring_tokens"]).replace("\n", "")
    # targets = " ".join(targets.strip().split())
    return clean_text(doc["docstring"])

def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    print(resps)
    return [[clean_text(s) for s in inner][0] for inner in resps]
