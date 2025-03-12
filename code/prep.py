r"""
Author: XUE Boyang      Filename: prep.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Data preparation: parse, preprocess, and save. 
Data output format: 
    {
        "question_id": "question_id",
        "question": "question",
        "answer": "answer"
    }
"""
import os
import argparse
import json

import datasets
import pandas as pd

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="webqa", choices=dataset_list)
parser.add_argument('--output_dir', type=str, default="./data/{}/raw")
args = parser.parse_args()


"""
TriviaQA dataset preparation and saving
"""
def prep_triviaqa_dataset(split="validation"):
    print(f'Preprocessing TriviaQA {split} dataset')
    data_pool = datasets.load_dataset("trivia_qa", "rc.nocontext", split=split)
    id_mem = set()

    def remove_dups(batch):
        if batch['question_id'][0] in id_mem:
            return {_:[] for _ in batch.keys()}
        id_mem.add(batch['question_id'][0])

        return batch

    data_pool = data_pool.map(remove_dups, batch_size=1, batched=True, 
                            load_from_cache_file=False, remove_columns=["search_results", "question_source", "entity_pages"])

    # Warrant the duplicated data was removed
    assert pd.Series([_['question_id'] for _ in data_pool]).value_counts().max() == 1

    data_set = []
    for data in data_pool:
        # import pdb; pdb.set_trace()
        instance = {
            "question_id": data["question_id"],
            "question": data["question"],
            "answer": data["answer"]["value"]
        }
        data_set.append(instance)
    
    print(f"Data size of {split}: {len(data_set)}")

    return data_set


def get_triviaqa_dataset(output_dir):
    # Get data splits
    data_splits = ["train", "validation", "test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_triviaqa_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
WebQA dataset preparation and saving
"""
def prep_webqa_dataset(data_path, split="validation"):
    print(f'Preprocessing WebQA {split} dataset')
    data_pool = json.load(open(data_path, "r"))
    print(f"Original data size of {split}: {len(data_pool)}")

    # import pdb; pdb.set_trace()
    data_set = []
    for key, value in data_pool.items():
        answers = []
        for answer in value["evidences"].values():
            if answer["answer"][0] != "no_answer":
                # import pdb; pdb.set_trace()
                answers.append(answer["answer"][0])

        # import pdb; pdb.set_trace()
        # print(answers)
        if answers:
            instance = {
                "question_id": key,
                "question": value["question"],
                "answer": max(answers, key=answers.count)
            }
            data_set.append(instance)
    
    print(f"Processed data size of {split}: {len(data_set)}")

    return data_set


def get_webqa_dataset(output_dir):
    # Get data splits
    data_splits = ["me_train", "me_validation.ir", "me_validation.ann", "me_test.ir", "me_test.ann"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_path = os.path.join("./../../data/WebQA.v1.0", f"{split}.json")
        data_set = prep_webqa_dataset(data_path, split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
GSM8K dataset preparation and saving
"""
def prep_gsm8k_dataset(split="test"):
    print(f'Preprocessing GSM8K {split} dataset')
    # Load GSM8K dataset
    dataset = datasets.load_dataset("gsm8k", "main", split=split)

    # data_set data questions and answers
    data_set = []
    for idx, item in enumerate(dataset):
        data_set.append({
            "question_id": idx,
            "question": item['question'],
            "answer": item['answer'] + "<|endoftext|>"
        })

    # Examples to construct few-shot examples
    if split == "train":
        examplars = {}
        for idx in range(0, 80, 8):
            examplars[f"examplar_{int(idx/8)+1}"] = data_set[idx:idx+8]
        
        prompt_path = "./data/gsm8k/few_shot_examplars.json"
        write_json(prompt_path, examplars)

    print(f"{len(data_set)} {split} examples")

    return data_set


def get_gsm8k_dataset(output_dir):
    # Get data splits
    data_splits = ["train", "test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_gsm8k_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)

"""
ASDIV dataset preparation and saving
"""
def prep_asdiv_dataset(split="test"):
    import xml.etree.ElementTree as ET
    import json

    # XML data as a string
    xml_data = '''<?xml version="1.0" encoding="UTF-8" ?>
    <Machine-Reading-Corpus-File>
        <ProblemSet>
            <Problem ID="nluds-0001" Grade="1" Source="http://www.k5learning.com">
                <Body>Seven red apples and two green apples are in the basket.</Body>
                <Question>How many apples are in the basket?</Question>
                <Solution-Type>Addition</Solution-Type>
                <Answer>9 (apples)</Answer>
                <Formula>7+2=9</Formula>
            </Problem>
        </ProblemSet>
    </Machine-Reading-Corpus-File>'''

    # Parse the XML
    root = ET.fromstring(xml_data)

    # Function to parse the XML and convert to dictionary
    def xml_to_dict(element):
        # Create a dictionary from the element attributes
        result = element.attrib
        
        # Iterate over child elements
        for child in element:
            # If the child has further children, treat it as a sub-dictionary
            if child:
                result[child.tag] = xml_to_dict(child)
            else:
                # Otherwise, just assign the text content
                result[child.tag] = child.text
        return result

    # Convert the entire XML to a dictionary
    data_dict = {root.tag: xml_to_dict(root)}

    # Convert dictionary to JSON
    json_data = json.dumps(data_dict, indent=4)

    # Print the JSON data
    print(json_data)


def get_asdiv_dataset(output_dir):
    # Get data splits
    data_splits = ["train", "test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_gsm8k_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
MATH dataset preparation and saving
"""
def prep_math_dataset(split="test"):
    print(f'Preprocessing MATH {split} dataset')
    dataset = datasets.load_dataset("lighteval/MATH", "all")
    # import pdb; pdb.set_trace()

    data_set = []

    for idx, data in enumerate(dataset[split]):
        # import pdb; pdb.set_trace()
        instance = {
            "question_id": str(idx+1),
            "level": data["level"].replace("Level ", ""),
            "type": data["type"],
            "question": data["problem"],
            "answer": data["solution"]
        }
        data_set.append(instance)
    
    print(f"Data size of {split}: {len(data_set)}")

    return data_set


def get_math_dataset(output_dir):
    data_splits = ["train", "test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_math_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
College dataset preparation and saving
"""
def prep_college_dataset(split="test"):
    print(f'Preprocessing College {split} dataset')
    input_path = f"./data/college/raw/full_{split}.json"
    dataset = read_jsonl(input_path)

    data_set, idx = [], 0

    for data in dataset:
        # import pdb; pdb.set_trace()
        # print(data["data_source"])
        if "college_math" in data["data_source"]:
            # import pdb; pdb.set_trace()
            instance = {
                "question_id": str(idx+1),
                "question": data["question"],
                "answer": data["answer"]
            }
            idx += 1
            data_set.append(instance)
    
    print(f"Data size of {split}: {len(data_set)}")

    return data_set


def get_college_dataset(output_dir):
    data_splits = ["train", "test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_college_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
TAL-SCQ dataset preparation and saving
"""
def prep_talscq_dataset(split="test"):
    print(f'Preprocessing TAL-SCQ {split} dataset')
    dataset = datasets.load_dataset("math-eval/TAL-SCQ5K", data_dir="TAL-SCQ5K-EN")
    # import pdb; pdb.set_trace()

    data_set = []

    for idx, data in enumerate(dataset[split]):
        # import pdb; pdb.set_trace()
        answer_dict = {}
        for answer in data["answer_option_list"]:
            answer_dict[answer[0]["aoVal"]] = answer[0]["content"]

        instance = {
            "question_id": data["qid"],
            "difficulty": data["difficulty"],
            "question": data["problem"],
            "answer": answer_dict[data["answer_value"]]
        }
        data_set.append(instance)
    
    print(f"Data size of {split}: {len(data_set)}")

    return data_set


def get_talscq_dataset(output_dir):
    data_splits = ["test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_talscq_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


"""
Theorem dataset preparation and saving
"""
def prep_theorem_dataset(split="test"):
    print(f'Preprocessing TheoremQA {split} dataset')
    dataset = datasets.load_dataset("TIGER-Lab/TheoremQA")
    # import pdb; pdb.set_trace()

    data_set = []

    for idx, data in enumerate(dataset[split]):
        import pdb; pdb.set_trace()
        instance = {
            "question_id": str(idx+1),
            "question": data["Question"],
            "answer": data["Answer"],
            "answer_type": data["Answer_type"],
            "Picture": False if data["Picture"] == None else True
        }
        data_set.append(instance)
    
    print(f"Data size of {split}: {len(data_set)}")

    return data_set


def get_theorem_dataset(output_dir):
    data_splits = ["test"]

    for split in data_splits:
        save_path = output_dir + f"/{split}.json"
        data_set = prep_theorem_dataset(split)

        with open(save_path, "w") as fw:
            json.dump(fp=fw, obj=data_set, indent=4, ensure_ascii=False)


if __name__=="__main__":
    # Output directory
    output_dir = args.output_dir.format(args.dataset)
    print(f"Data saved to {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset == "triviaqa":
        get_triviaqa_dataset(output_dir)
    elif args.dataset == "webqa":
        get_webqa_dataset(output_dir)
    elif args.dataset == "gsm8k":
        get_gsm8k_dataset(output_dir)
    elif args.dataset == "math":
        get_math_dataset(output_dir)
    elif args.dataset == "college":
        get_college_dataset(output_dir)
    elif args.dataset == "theorem":
        get_theorem_dataset(output_dir)
    elif args.dataset == "talscq":
        get_talscq_dataset(output_dir)

