r"""
Author: XUE Boyang      Filename: infer.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Inference decoding on test sets.
"""
import os
import time
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from tqdm import tqdm
import json
import random

import torch
import transformers
from peft import PeftModel
from torch import distributed as dist
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import set_seed, GenerationConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams

from utils import *


@dataclass
class ModelArguments:
    model_type: str = field(default="vanilla", metadata={"help": "Model type.", "choices": ["vanilla", "tune"]})
    model_name: str = field(default="llama31_ins", metadata={"help": "Model name.", "choices": model_path_dict.keys()})
    train_path: str = field(default="./exp/{}/train", metadata={"help": "Directory to save the training results."})
    save_suffix: str = field(default="base_train", metadata={"help": "Suffix of the saved model."})
    # LoRA setting
    lora_use: bool = field(default=False, metadata={"help": "Use LoRA or not."})
    lora_weights: str = field(default="./exp/{}/train", metadata={"help": "LoRA weights path."})
    model_suffix: str = field(default="no_lora", metadata={"help": "File name to save the results."})
    # Bits and Bytes config
    bnb_use: bool = field(default=False, metadata={"help": "Whether to use BitsAndBytesConfig."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to enable 4-bit quantization."})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Set the quantization data type in the bnb.nn.Linear4Bit layers", "choices": ["fp4", "nf4"]})
    bnb_4bit_compute_dtype: torch.dtype = field(default=torch.float16, metadata={"help": "Set the computational type which might be different than the input type."})
    # Tokenizer setting
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    use_fast: bool = field(default=False, metadata={"help": "Whether to use rust based fast tokenizer."})
    padding_side: str = field(default="right", metadata={"help": "Padding side for sequences with different length."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files."})
    # vLLM setting
    vllm_use: bool = field(default=False, metadata={"help": "Use vLLM or not."})
    gpu_memory_utilization: float = field(default=0.65, metadata={"help": "GPU memory utilization."})


@dataclass
class DataArguments:
    data_dir: str = field(default="./data/{}/raw", metadata={"help": "Directory to save data."})
    dataset: str = field(default="triviaqa", metadata={"help": "Dataset name.", "choices": dataset_list})
    data_file: str = field(default="test", metadata={"help": "Data file name."})
    prompt_dir: str = field(default="./prompt/", metadata={"help": "Path to the prompt."})
    continue_generate: bool = field(default=False, metadata={"help": "Continue from the previous generations."})
    split_id: int = field(default=0, metadata={"help": "Split id for inference."})
    split_num: int = field(default=1, metadata={"help": "Number if splited subsets."})


@dataclass
class InferenceArguments:
    icl_use: bool = field(default=False, metadata={"help": "Use few-shot prompt or not."})
    do_sample: bool = field(default=False, metadata={"help": "Whether to use sampling or not."})
    output_dir: str = field(default="./exp/{}/infer", metadata={"help": "Directory to save results."})
    num_sampling: int = field(default=5, metadata={"help": "Number of samples."})
    temperature: float = field(default=0.8, metadata={"help": "Temperature for sampling."})
    top_p: float = field(default=1.0, metadata={"help": "Top p for sampling."})
    top_k: int = field(default=40, metadata={"help": "Top k for sampling."})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for sampling."})
    max_length: int = field(default=16, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    repetition_penalty: float = field(default=1.1, metadata={"help": "Repetition penalty."})


@dataclass
class DeviceArguments:
    device: str = field(default="cuda", metadata={"help": "Device to use."})
    seed: int = field(default=3407, metadata={"help": "Random seed."})
    gpu_num: int = field(default=1, metadata={"help": "Number of GPUs."})
    local_rank: int = field(default=0, metadata={"help": "Local rank."})
    global_rank: int = field(default=0, metadata={"help": "Global rank."})
    world_size: int = field(default=0, metadata={"help": "World size."})


# Parse arguments.
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments, DeviceArguments))
model_args, data_args, infer_args, device_args = parser.parse_args_into_dataclasses()


# Resize tokenizer and embedding.
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# Format the few-shot examplar of list to string.
def format_examplar(few_shot_examplars, examplar_split):
    few_shot_examplar_list = []
    for few_shot_examplar in few_shot_examplars.values():
        few_shot_examplas = []
        for few_shot_example in few_shot_examplar:
            few_shot_examplas.append("{}{}\n{}{}".format(examplar_split["input"], few_shot_example["question"], 
                                                         examplar_split["output"], few_shot_example["answer"]))
        few_shot_examplar_list.append("\n\n".join(few_shot_examplas))

    return few_shot_examplar_list


# Split the generation to get the answer part.
def output_split(output, tokenizer, split_len, prompt_split, vllm_use):
    if vllm_use:
        text = output[0].outputs[0].text
        return text, text.split(prompt_split)[0]
    else:
        text = tokenizer.decode(output.sequences[0][split_len:], skip_special_tokens=True)
        return text, text.split(prompt_split)[0]


def infer():
    # import pdb; pdb.set_trace()
    # Info: Device settings: random seed, using cuda or not, distributed setting.
    set_seed(device_args.seed)

    device_args.num_gpu = torch.cuda.device_count()
    device_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up logging.
    infer_args.output_dir = os.path.join(infer_args.output_dir.format(data_args.dataset), 
                                         "{}_{}_{}{}{}".format(model_args.model_name, 
                                                              data_args.dataset, 
                                                              model_args.model_suffix,
                                                              "_vllm" if model_args.vllm_use else "",
                                                              "_icl" if infer_args.icl_use else ""))

    if not os.path.exists(infer_args.output_dir):
        os.makedirs(infer_args.output_dir)

    infer_args.log_path = os.path.join(infer_args.output_dir, f"{data_args.data_file}_generate.log" \
                                      if data_args.split_num == 0 else f"{data_args.data_file}_generate_{data_args.split_id}.log")

    logging.basicConfig(
        filename=infer_args.log_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    # Load model and tokenizer.
    if model_args.bnb_use:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=model_args.bnb_4bit_compute_dtype
        )
        logging.info(f"BitsAndBytes use.")
    else:
        logging.info(f"No BitsAndBytes use.")

    # Parse the model name or path.
    if model_args.model_type == "vanilla":
        model_name_or_path = model_path_dict[model_args.model_name]
    elif model_args.model_type == "tune":
        model_name_or_path = os.path.join(model_args.train_path.format(data_args.dataset), 
                                          f"{model_args.model_name}_{data_args.dataset}_{model_args.save_suffix}")

    logging.info(f"Loading model and tokenizer from {model_name_or_path} ...")
    if model_args.vllm_use:
        model = LLM(model=model_name_or_path,
                    dtype=torch.float16,
                    max_model_len=model_args.model_max_length,
                    # quantization=bnb_config if model_args.bnb_use else None,
                    gpu_memory_utilization=model_args.gpu_memory_utilization)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config if model_args.bnb_use else None,
            device_map="balanced" # device_map: "auto", "balanced", "balanced_low_0", "sequential"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        model_max_length=model_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=model_args.use_fast
    )

    # Resize tokenizer and embedding.
    # import pdb; pdb.set_trace()
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})

    if model_args.lora_use:
        model_args.lora_weights = os.path.join(model_args.lora_weights.format(data_args.dataset), 
                                         f"{model_args.model_name}_{data_args.dataset}_{model_args.model_suffix}")
        model = PeftModel.from_pretrained(
            model,
            model_id=model_args.lora_weights,
            torch_dtype=torch.float16
        )

    # print(device)
    # Load data.
    # import pdb; pdb.set_trace()
    data_path = os.path.join(data_args.data_dir.format(data_args.dataset), 
                                  "{}.json".format(data_args.data_file))

    logging.info(f"Loading data from {data_path} ...")
    dataset = json.load(open(data_path))

    # Load prompt and select the prompt type.
    prompt_template = json.load(open(os.path.join(data_args.prompt_dir, 
                                                  f"{data_args.dataset}_template.json")))
    instruction = prompt_template["instruction"]
    prompt_split = prompt_template["output_split"]
    if infer_args.icl_use:
        few_shot_examplars, examplar_split = prompt_template["generate_few_shot_examplar"], prompt_template["few_shot_split"]
        few_shot_examplar_list = format_examplar(few_shot_examplars, examplar_split)
        prompt_input = prompt_template["few_shot_prompt"]
    else:
        prompt_input = prompt_template["standard_prompt"]

    # Format the output file.
    # import pdb; pdb.set_trace()
    infer_args.save_path = os.path.join(infer_args.output_dir, f"{data_args.data_file}_generate.json" \
                                        if data_args.split_num == 0 else f"{data_args.data_file}_generate_{data_args.split_id}.json")
    if data_args.continue_generate:
        exist_num = len(read_jsonl(infer_args.save_path))
        # Split the dataset if needed.
        dataset = dataset[exist_num::]
    else:
        open(infer_args.save_path, "w").close()

    # Split the dataset if needed.
    if data_args.split_num > 1:
        data_len = len(dataset)
        split_len = (data_len // data_args.split_num) + 1
        dataset = dataset[data_args.split_id * split_len:(data_args.split_id + 1) * split_len]
        logging.info(f"Data length: {data_len}; Start from {dataset[0]['question_id']}; End at {dataset[-1]['question_id']}")

    data_len = len(dataset)

    logging.info(f"Arguments:\nModel Arguments: {model_args}\nData Arguments: {data_args}\nInference Arguments: {infer_args}")
    logging.info(f"The number of dataset: {data_len}")

    # Sample the data.
    start_time = time.time()
    logging.info("Start generating ...")
    first_log_flag = True

    # Set up the decoding configuration.
    if model_args.vllm_use:
        temperature = infer_args.temperature if infer_args.do_sample else 0.0
        # print(temperature)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=infer_args.top_p,
            top_k=infer_args.top_k,
            max_tokens=infer_args.max_length,
            repetition_penalty=infer_args.repetition_penalty
        )
        logging.info(sampling_params)
    else:
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=infer_args.temperature,
            top_p=infer_args.top_p,
            top_k=infer_args.top_k,
            num_beams=infer_args.num_beams,
            repetition_penalty=infer_args.repetition_penalty
        ) if infer_args.do_sample \
        else GenerationConfig(
            do_sample=False,
            num_beams=infer_args.num_beams,
            repetition_penalty=infer_args.repetition_penalty)
        logging.info(generation_config)
        
    with tqdm(total=data_len) as t:
        for idx, batch in enumerate(dataset):
            if infer_args.icl_use:
                few_shot_examplar = random.choice(few_shot_examplar_list)
                input_tokens = prompt_input.format(instruction=instruction, examples=few_shot_examplar, question=batch["question"])
            else:
                input_tokens = prompt_input.format(instruction=instruction, question=batch["question"])
    
            with torch.no_grad():
                if model_args.vllm_use:
                    # import pdb; pdb.set_trace()
                    generation = model.generate(input_tokens, sampling_params)
                    output_sequence, generated_text = output_split(generation, None, None, prompt_split, model_args.vllm_use)
                else:
                    # import pdb; pdb.set_trace()
                    # time.sleep(1)
                    input_ids = tokenizer(input_tokens, padding=True, return_tensors="pt")["input_ids"].to(device_args.device)
                    # print(input_ids.size())

                    # print(model.module.device)
                    # print(input_ids.device)

                    generation = model.generate(input_ids,
                                            generation_config=generation_config,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            max_new_tokens=infer_args.max_length,
                                            pad_token_id=tokenizer.pad_token_id,
                                            eos_token_id=tokenizer.eos_token_id, 
                                            bos_token_id=tokenizer.bos_token_id)

                    # import pdb; pdb.set_trace()
                    output_sequence, generated_text = output_split(generation, tokenizer, len(input_ids[0]), prompt_split, model_args.vllm_use)
                    pass
                pass

            if first_log_flag:
                logging.info("Input Prompt: \n{}".format(prompt_input.format(instruction=instruction, examples=few_shot_examplar, question=batch["question"]) \
                                                            if infer_args.icl_use else prompt_input.format(instruction=instruction, question=batch["question"])))
                logging.info("LLM Generation: \n{}".format(generation))
                first_log_flag = False

            # print(data_point)
            instance = {
                "question_id": batch["question_id"] if "question_id" in batch.keys() else f"id_{idx+1}",
                "question": batch["question"],
                "answer": batch["answer"],
                "generation": output_sequence,
                "output": generated_text
            }

            # print(instance)

            # Real-time saving the results.
            with open(infer_args.save_path, "a+") as fw: 
                instance_write = json.dumps(obj=instance, ensure_ascii=False)
                fw.write(instance_write + '\n')

            t.set_postfix()
            t.update(1)
    
    # import pdb; pdb.set_trace()
    elapsed_time = format_seconds(time.time() - start_time)
    logging.info(f"Total elapsed time: {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s")

    # Convert jsonl to json format.
    logging.info("Generating is done.")
    # jsonl2json(infer_args.save_path, infer_args.save_path)
    logging.info(f"Save to {infer_args.save_path}")


if __name__ == "__main__":
    infer()

