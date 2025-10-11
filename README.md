# BACE: Unified Benchmarking Accuracy and Energy of Code Language Models


## Overview
This repository is a forked repository from [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness). All functionalities of the mentioned repository exists here as well.

We extended that repository with the aim of capturing energy consumption of language models across different benchmarks. Also, we proposed two new methods of scoring LLMs `Observation-To-Expectation (OTE)` and `Concentric-Incremental-Circles (CIRC)` that ranks LLM on 1-5 scale.

## Install
```
git clone https://github.com/SMART-Dal/LLM-Sustainability-Rater.git
cd LLM-Sustainability-Rater
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage Steps
- First you need to create `.env` file with your Huggingface API token in this format `HF_TOKEN=YOUR_TOKEN`.
- `cd lm_eval`
- Then you need to configure `run_config.yaml` file to specify the models and benchmarks.
    - `model`: This tag should be set to `hf` because our main goal is to run base models without any optimizations
    - `tasks`: You can extract the list of tasks with `lm-eval --tasks list` command.
    - `model_args`: Any argument that `AutoModel.from_pretrained` of huggingface takes.
    - `bnb_config`: for each model you define in `model_args`, you should specify a bnb tag that lets model know if it needs to quantize the model with bitsandbytes quantization. For instance, you can set `load_in_4bit=True`. The `model_args` and `bnb_config` lists are index-aligned, such that each model configuration corresponds to its respective quantization setting.
    - `experiments_run`: Simple tag to determine the purpose of your project. It will be stored in the final results file so you can extract and filter your results.
- Then you can run the command the `python main_run.py`. Each model specified in the yaml file will run sequentially on one task a time.
- The final results will be stored in `results/results.jsonl`. Also, if you need to take a look at each model specific energy log by codecarbon, you can view `codecarbon_log/{task_name}/{model_name}/{model_task}.log`
- After generating results, you can rate your LLMs on a benchmark by running `python rating_llms/{approach}.py --task_name {task_name} --file_name {file_name}` where `task_name` is the tasks you specified in `run_config.yaml` and `file_name` will point to default generated `results.jsonl` file unless you specify. The results of ranking will be stored in `rating_llm/data` 



### Experimented Models for UBACE:
- [deepseek-ai/deepseek-coder-6.7b-base](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base)
- [bigcode/starcoderbase-1b](https://huggingface.co/bigcode/starcoderbase-1b)
- [bigcode/starcoder2-3b](https://huggingface.co/bigcode/starcoder2-3b)
- [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
- [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
- [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B)
- [deepseek-ai/deepseek-coder-1.3b-base](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)
- [deepseek-ai/deepseek-coder-7b-instruct-v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
- [bigcode/starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b)
- [Salesforce/codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono)
- [Qwen/Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
- [01-ai/Yi-Coder-9B](https://huggingface.co/01-ai/Yi-Coder-9B)
- [rombodawg/rombos_Replete-Coder-Llama3-8B](https://huggingface.co/rombodawg/rombos_Replete-Coder-Llama3-8B)
- [uukuguy/speechless-code-mistral-7b-v1.0](https://huggingface.co/uukuguy/speechless-code-mistral-7b-v1.0)
- [stabilityai/stable-code-3b](https://huggingface.co/stabilityai/stable-code-3b)
- [codellama/CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)
- [ByteDance-Seed/Seed-Coder-8B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct)
- [Qwen/CodeQwen1.5-7B-Chat](https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat)
- [ise-uiuc/Magicoder-S-DS-6.7B](https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B)
- [ibm-granite/granite-8b-code-base-4k](https://huggingface.co/ibm-granite/granite-8b-code-base-4k)
- [Salesforce/codegen2-7B_P](https://huggingface.co/Salesforce/codegen2-7B_P)
- [Salesforce/codegen-2B-mono](https://huggingface.co/Salesforce/codegen-2B-mono)