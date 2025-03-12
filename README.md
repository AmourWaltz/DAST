# DAST: Difficulty-Aware Self-Training for Large Language Models

This project and related paper is in progress.

## Implementation

### Data construction and split the difficulty levels

```shell
python code/sample.py --model_name llama31 --dataset triviaqa --data_file test --model_suffix base
python code/split.py --model_name llama31 --dataset gsm8k --data_file train --suffix 8k_8s

```


### LLM Training, Inference, and Evaluation

```shell
# model: [llama31 llama31_ins qwen25_ins qwen25 mistral]
# data: [gsm8k 200, college 200, math 512, theorem 100, talscq 48]

CUDA_VISIBLE_DEVICES=0,1,2,3 python code/train_sft.py --model_name llama31 --dataset gsm8k --data_file train --save_suffix base --lora_use true
CUDA_VISIBLE_DEVICES=0 python code/infer.py --model_name llama31 --dataset gsm8k --model_type tune --data_file test --model_suffix base_sft --save_suffix train_base --max_length 200
python code/eval.py --model_name $model --dataset $data --data_file test --model_suffix base_vllm_icl


```
