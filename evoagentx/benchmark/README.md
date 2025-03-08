# Benchmark 

## Benchmark Overview 

This repository provides a set of benchmarks to facilitate the evaluation of different agent-based systems. Below is a summary of the benchmarks currently included, along with basic dataset statistics: 


| Task                      | Dataset Name    | # Train   | # Dev   | # Test |
| ------------------------- | --------------- | --------- | ------- | ------ |
| QA                        | NQ              | 79,168    | 8,757   | 3,610  |
| Multi-Hop QA              | HotPotQA        | 90,447    | 7,405   | /      |
| Math                      | GSM8K           | 7,473     | /       | 1,319  |
| Math                      | MATH            | /         | /       | /      |
| Code Generation           | HumanEval       | /         | /       | /      |
| Code Generation           | MBPP            | /         | /       | /      |
| Code Generation           | LiveCodeBench   | /         | /       | /      |


Below, we introduce the preprocessing steps and evaluation metrics for each benchmark. 

- [Question Answering](#question-answering)
  - [NQ](#nq)
  - [HotPotQA](#hotpotqa)
- [Math](#math)
  - [GSM8K](#gsm8k)
  - [MATH](#math)
- [Code Generation](#code-generation)
  - [HumanEval](#humaneval)
  - [MBPP](#mbpp)
  - [LiveCodeBench](#livecodebench)

## Preprocessing and Evaluation Metrics 

### Question Answering 

For the QA datasets, we use Exact Match (EM), F1, and Accuracy (ACC) as evaluation metrics. EM requires the predicted answer to be exactly the same as the ground truth answer. ACC requires that the predicted answer contains the ground-truth answer, which is usefule when the LLM is used to generate the answer. 

#### NQ
[Natural Questions (NQ)](https://github.com/google-research-datasets/natural-questions) contains questions from the Google search engine and the answers, annotated by human annotators, are paragraphs or entities in the Wikipedia page of the top 5 search results. We use the dataset splits provided by the [DPR](https://github.com/facebookresearch/DPR) repository, which contains 79,168 training, 8,757 development, and 3,610 test examples. 

You can load the dataset using the following code: 
```python
from evoagentx.benchmark import NQ
nq_dataset = NQ() # optional: path="/path/to/save_data"
test_data = nq_dataset.get_test_data()
```
Each example in the dataset is in the following format: 
```json
{
    "id": "test-1", 
    "question": "the question", 
    "answers": ["possible answers"]
}
```


#### HotPotQA 
[HotPotQA](https://hotpotqa.github.io/) is a multi-hop QA dataset that requires multi-step reasoning to answer the question. We use the distractor setting of the dataset. Each example contains a question, an answer, some context that contians both supporting and distractor information, and supporting facts. We only include the training and development sets, as the test set is not publicly available. 

You can load the dataset using the following code: 
```python
from evoagentx.benchmark import HotPotQA
hotpotqa_dataset = HotPotQA() # optional: path="/path/to/save_data"
test_data = hotpotqa_dataset.get_test_data()
```
Each example in the dataset is in the following format, where the second element (int) of a supporting_fact is the index of the sentence in the context that supports the answer. 
```json
{
        "_id": "the id of the example", 
        "question": "the question", 
        "answer": "the answer", 
        "context": [["context_title", ["context_sentence", "another_sentence"]]],
        "supporting_facts": [["supporting_title", 0]]
    }
```


### Math

#### GSM8K 
[GSM8K](https://github.com/openai/grade-school-math) consists of high quality grade school math problems created by human problem writers. These problems require multi-step mathematical reasoning to solve. We use the dataset splits provided by the original repository, which contains 7.5K training problems and 1K test problems. 

You can load the dataset using the following code: 
```python
from evoagentx.benchmark import GSM8K
gsm8k_dataset = GSM8K() # optional: path="/path/to/save_data"
test_data = gsm8k_dataset.get_test_data()
```
Each example in the dataset is in the following format: 
```json
{
    "id": "test-1", 
    "question": "the question", 
    "answer": "the answer"
}
```

#### MATH 

### Code Generation 


HotPotQA is a 





#### HumanEval 

#### MBPP 

#### LiveCodeBench 

