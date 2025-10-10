# BACE: Unified Benchmarking Accuracy and Energy of Code Language Models


## Overview
This repository is a forked repository from [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness). All functionalities of the mentioned repository exists here as well.

We extended that repository with the aim of capturing energy consumption of language models across different benchmarks. Also, we proposed two new methods of scoring LLMs `Observation-To-Expectation (OTE)` and `Concentric-Incremental-Circles (CIC)` that ranks LLM on 1-5 scale.

## Install
```
git clone https://github.com/SMART-Dal/LLM-Sustainability-Rater.git
cd LLM-Sustainability-Rater
pip install -e .
cd lm_eval
```

## Usage
- First you need to configure `run_config.yaml` file to specify the models and benchmarks.
    - `model`: This tag should be set to `hf` because our main goal is to run base models without any optimizations
    - `tasks`: You can extract the list of tasks with `lm-eval --tasks list` command.
    - `model_args`: Any argument that `AutoModel.from_pretrained` of huggingface takes.
    - `bnb_config`: for each model you define in `model_args`, you should specify a bnb tag that lets model know if it needs to quantize the model with bitsandbytes quantization. For instance, you can set `load_in_4bit=True`. The `model_args` and `bnb_config` lists are index-aligned, such that each model configuration corresponds to its respective quantization setting.
    - `experiments_run`: Simple tag to determine the purpose of your project. It will be stored in the final results file so you can extract and filter your results.
- Then you can run the command the `python main_run.py`. Each model specified in the yaml file will run sequentially on one task a time.
- The final results will be stored in `results/results.jsonl`. Also, if you need to take a look at each model specific energy log by codecarbon, you can view `codecarbon_log/{task_name}/{model_name}/{model_task}.log`