import re
import datasets
import logging

# output_logger = logging.getLogger("generation_logger")

def process_docs(dataset: datasets.Dataset):
    dataset = dataset.select([i for i in range(500)])
    return dataset

def doc_to_text(doc):
    inputs = " ".join(doc["code_tokens"]).replace("\n", " ")
    inputs = " ".join(inputs.strip().split())
    prompt = """Generate a Python docstring for the following code. The docstring should follow best practices and clearly explain the code's purpose, its parameters (inputs), return value (output). Remember to only generate docstring, and DO NOT generate method or class header.\n\nMain code:\n\n```python\n{}```\n\nYour generated docstring:"""
    return prompt.format(inputs)
    

def clean_text(text):
    text = text.replace('\n', ' ').lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def doc_to_target(doc):
    # targets = " ".join(doc["docstring_tokens"]).replace("\n", "")
    # targets = " ".join(targets.strip().split())
    return clean_text(doc["docstring"])

def parse_output(text: str) -> str:
    start_idx = text.find("\"""")
    if start_idx == -1:
        return text
    end_idx = text.find("\"""", start_idx + 3)
    if end_idx == -1:
        return text
    
    return text[start_idx + 3: end_idx]

def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    new_res = [parse_output([clean_text(s) for s in inner][0]) for inner in resps]
     
    # for smp in new_res:
    #     output_logger.info(smp)
    # output_logger.info("-" * 30)
    return new_res
