r"""
Author: XUE Boyang      Filename: split.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Split the knowledge boundary information based on the generated samples.
"""
import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
# from ipdb import set_trace

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str, default="prep", help="Different stages.", choices=["prep", "dpo"])
parser.add_argument('--model_name', type=str, default="llama3", help="Model name.", choices=model_path_dict.keys())
parser.add_argument('--dataset', type=str, default="gsm8k", help="Dataset name.", choices=dataset_list)
parser.add_argument('--data_file', type=str, default="train", help="Data file name.", choices=["train", "test"])
parser.add_argument('--model_suffix', type=str, default="train_sft_base_bnb_icl", help="Model suffix.")
parser.add_argument('--data_suffix', type=str, default="sample_8shot", help="File name to save the results.")
parser.add_argument('--input_dir', type=str, default="./data/{}/prep/")
parser.add_argument('--output_dir', type=str, default="./data/{}/split/")
parser.add_argument('--split_num', type=int, default=0, help="Number of splits.")
parser.add_argument('--icl_use', type=bool, default=False, help="Use few-shot prompt or not.")
parser.add_argument('--vllm_use', type=bool, default=False, help="Use few-shot prompt or not.")
parser.add_argument('--split_id', type=int, default=0, help="Split ID.")
parser.add_argument('--max_length', type=int, default=1024, help="Max length of the input.")

args = parser.parse_args()


def draw_scores_histogram(greedy_scores, sample_scores, output_dir):
    # Plot the distribution of the samples
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(greedy_scores, bins=5, color='green')
    plt.title('Distribution of Greedy Decoding Scores')
    plt.xlabel('Greedy Scores')
    plt.ylabel('Number of Samples')
    plt.grid(False)
    plt.legend(handles=[plt.Line2D([0], [0], color='green', label='Greedy Scores')])

    plt.subplot(1, 2, 2)
    plt.hist(sample_scores, bins=20, color='blue')
    plt.title('Distribution of Temperature Sampling Scores')
    plt.xlabel('Sampling Scores')
    plt.ylabel(f'Number of Samples (N={len(sample_scores)})')
    plt.grid(False)
    plt.legend(handles=[plt.Line2D([0], [0], color='blue', label='Greedy Scores')])

    plt.tight_layout() 
    plt.savefig(f"{output_dir}/scores_histogram.png", dpi=600)
    

def draw_known_pie(categories, output_dir):
    labels = categories.keys()
    sizes = categories.values()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff00ff', '#c2c2f0']
    explode = (0, 0, 0, 0, 0, 0) 

    plt.figure(figsize=(8, 4))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=0)
    plt.title(f'Partitions of Samples in Different Difficulty Levels on {args.dataset} {args.data_file} set', fontsize=10)

    plt.axis('equal')
    plt.savefig(f"{output_dir}/known_level_pie.png", dpi=600)


def known_level(greedy_score, sample_score, dataset):
    if dataset == "triviaqa":
        if greedy_score == 1:
            return Levels[1]
        elif 0.75 < greedy_score < 1:
            return Levels[2]
        elif 0.35 < greedy_score <= 0.75:
            return Levels[3]
        elif 0 < greedy_score <= 0.35:
            return Levels[4]
    elif dataset == "webqa":
        if greedy_score == 1:
            return Levels[1]
        elif 0.70 < greedy_score < 1:
            return Levels[2]
        elif 0.30 < greedy_score <= 0.70:
            return Levels[3]
        elif 0 < greedy_score <= 0.30:
            return Levels[4]
    elif dataset == "gsm8k":
        if greedy_score == 1:
            return Levels[1]
        elif 0.70 < greedy_score < 1:
            return Levels[2]
        elif 0.40 <= greedy_score <= 0.70:
            return Levels[3]
        elif 0.10 < greedy_score < 0.40:
            return Levels[4]
        elif 0 < greedy_score <= 0.10:
            return Levels[5]

    if sample_score > 0:
        return Levels[5]
    else:
        return Levels[6]


def category_calculate(greedy_scores, sample_scores, dataset):
    num_certain, num_high, num_medium, num_low, num_weak, num_unknown = 0, 0, 0, 0, 0, 0
    
    """
    Split strategies for different datasets.
    """
    for greedy_score, sample_score in zip(greedy_scores, sample_scores):
        data_known = known_level(greedy_score, sample_score, dataset)
        # print(data_known)
        if data_known == Levels[1]:
            num_certain += 1
            continue
        elif data_known == Levels[2]:
            num_high += 1
            continue
        elif data_known == Levels[3]:
            num_medium += 1
            continue
        elif data_known == Levels[4]:
            num_low += 1
            continue
        elif data_known == Levels[5]:
            num_weak += 1
            continue
        elif data_known == Levels[6]:
            num_unknown += 1
            continue
        else:
            raise ValueError(f"Unknown data known level: {data_known}")
    
    # print("len(greedy_scores): ", len(sample_scores))
    # print(f"num_certain: {num_certain}, num_high: {num_high}, num_medium: {num_medium}, num_low: {num_low}, num_weak: {num_weak}, num_unknown: {num_unknown}")
    assert num_certain + num_high + num_medium + num_low + num_weak + num_unknown == len(greedy_scores)

    categories = {
        Levels[1]: num_certain,
        Levels[2]: num_high,
        Levels[3]: num_medium,
        Levels[4]: num_low,
        Levels[5]: num_weak,
        Levels[6]: num_unknown
    }

    return categories    


def data_split_prep():
    # Format output file.
    args.output_dir = args.output_dir.format(args.dataset)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.output_dir = os.path.join(args.output_dir, "{}_{}_{}_{}". \
                                   format(args.model_name, args.dataset, args.data_file, args.suffix))
    log_path = f"{args.output_dir}/split.log"
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    

    # Loading dataset.
    data_path = os.path.join(args.input_dir.format(args.dataset), 
                                  "{}_{}_{}_{}.json".format(args.model_name, args.dataset, args.data_file, args.suffix))
    data_pool = read_json(data_path)
    
    # Set logging.
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    logging.info(f"Load data from {data_path} ...")

    greedy_scores, sample_scores = [], []
    for data in data_pool:
        score = data["scores"]
        greedy_scores.append(score["greedy_scores_avg"])
        if score["sample_scores"]:
            sample_scores.append(score["sample_scores_avg"])
        else:
            sample_scores.append(0)

    # Draw histogram.
    draw_scores_histogram(greedy_scores, sample_scores, args.output_dir)
    categories = category_calculate(greedy_scores, sample_scores, args.dataset)
    logging.info(categories)
    draw_known_pie(categories, args.output_dir)


def dpo_data_prep():
    args.output_dir = "./data/{}/prep/"

    args.output_dir = os.path.join(args.output_dir.format(args.dataset), "{}_{}_{}_{}". \
                                   format(args.model_name, args.dataset, args.data_file, args.suffix))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


if __name__ == '__main__':
    if args.stage == "prep":
        data_split_prep()
    elif args.stage == "dpo":
        dpo_data_prep()
