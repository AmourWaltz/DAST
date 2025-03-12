r"""
Author: XUE Boyang      Filename: eval.py
Afflition: MoE Key Lab, The Chinese University of Hong Kong.
Description: Evaluation scripts on QA and Math datasets.
"""
import logging
import re
import os
import string
import argparse

from utils import *
from split import known_level, Levels

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default="llama31_ins", help="Model name.", choices=model_path_dict.keys())
parser.add_argument('--model_suffix', type=str, default="train_sft_base_bnb_icl", help="Model suffix.")
parser.add_argument('--dataset', type=str, default="gsm8k", help="Dataset name.", choices=dataset_list)
parser.add_argument('--data_file', type=str, default="test", help="Data file name.")
parser.add_argument('--data_suffix', type=str, default="1k_8s", help="File name to save the results.")
parser.add_argument('--score_dir', type=str, default="./data/{}/prep/")
parser.add_argument('--input_dir', type=str, default="./exp/{}/infer/")
parser.add_argument('--score_use', type=bool, default=False, help="Whether to use the score file.")
parser.add_argument('--split_num', type=int, default=0, help="Number of splits.")
parser.add_argument('--icl_use', type=bool, default=False, help="Use few-shot prompt or not.")
parser.add_argument('--vllm_use', type=bool, default=False, help="Use few-shot prompt or not.")
parser.add_argument('--split_id', type=int, default=0, help="Split ID.")
parser.add_argument('--max_length', type=int, default=1024, help="Max length of the input.")

args = parser.parse_args()


"""
Evaluate QA outputs: TriviaQA, WebQA
"""
# Normalize the answer.
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    # return white_space_fix(remove_punc(lower(s)))
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Compute the exact match score in different ways.
def compute_exact(a_gold, a_pred):
    eval_type = "EM_RP"

    if eval_type == "EM":
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))
    elif eval_type == "EM_R":
        return int(normalize_answer(a_gold) in normalize_answer(a_pred))
    elif eval_type == "EM_P":
        return int(normalize_answer(a_pred) in normalize_answer(a_gold))
    elif eval_type == "EM_RP":
        if args.dataset == "triviaqa":
            return int(normalize_answer(a_gold) in normalize_answer(a_pred)) or int(normalize_answer(a_pred) in normalize_answer(a_gold))
        elif args.dataset == "gsm8k":
            return int(normalize_answer(a_gold) == normalize_answer(a_pred))


"""
Evaluate math problems: CollegeMath
"""
# Copyright from mathscale/MWPBench/eval_vllm/util.py https://github.com/microsoft/unilm/blob/master/mathscale/MWPBench/eval_vllm/util.py
def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    # print(substrs)
    # print(new_str)
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr == "":
                continue

            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception as e:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def unbox_and_extract(text):
    start_indices = [m.start() for m in re.finditer(r'\\boxed{', text)]
    extracted_contents = []
    for start in start_indices:
        brace_count = 0
        for i, char in enumerate(text[start:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    extracted_contents.append(text[start+7:end-1])  # +7 to skip '\\boxed{'
                    break
    # Replace '\\boxed{...}' with the content inside it
    unboxed_text = re.sub(r'\\boxed{(.*?)}', r'\1', text)
    return unboxed_text, extracted_contents


def convert_to_latex_fraction(text: str) -> str:
    # Use regex to find all occurrences of ((num)/(denom))
    pattern = re.compile(r"\(\(([\d]+)\)/\(([\d]+)\)\)")
    
    matches = pattern.findall(text)
    
    for match in matches:
        num, denom = match
        latex_frac = f"\\\\frac{{{num}}}{{{denom}}}"
        
        # Replace the old expression with the LaTeX fraction
        text = text.replace(f"(({num})/({denom}))", latex_frac)
    
    return text


def strip_string(string):
    # convert ((3)/(4)) -> \\frac{3}{4}
    string = convert_to_latex_fraction(string)

    # remove ,
    string = string.replace(",", "")

    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # My own
    string = string.replace("\\quad", " ")
    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    # print(string)
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_number(s):
    s = s.strip("$")

    try:
        # Try to convert the string to an integer
        int(s)
        return True
    except ValueError:
        try:
            # Try to convert the string to a float
            float(s)
            return True
        except ValueError:
            return False


def is_single_inline_math(expression: str) -> bool:
    # Use regex to check for a pattern that starts and ends with dollar signs,
    # and contains no other dollar signs in between.
    pattern = re.compile(r"^\$[^$]+\$$")
    
    match = pattern.match(expression)
    
    return bool(match)


def is_equiv(prediction_ans, reference_ans, verbose=False):
    if prediction_ans is None and reference_ans is None:
        print("WARNING: Both None")
        return True, prediction_ans, reference_ans
    if prediction_ans is None or reference_ans is None:
        return False, prediction_ans, reference_ans

    try:
        clean_prediction_ans = strip_string(prediction_ans)
        clean_reference_ans = strip_string(reference_ans)

        if is_number(clean_prediction_ans) and is_number(clean_reference_ans):
            judge = float(clean_prediction_ans.strip("$")) == float(clean_reference_ans.strip("$"))
            # print(f"1 judge: {judge}")
        elif is_single_inline_math(clean_reference_ans):
            judge = (clean_reference_ans.strip("$") in clean_prediction_ans.strip("$"))
            # print(f"2 judge: {judge}")
        elif (len(clean_prediction_ans) >= 3) and (not is_number(clean_prediction_ans)) and (not clean_prediction_ans.startswith("-")) and (not clean_reference_ans.startswith("-")) and (clean_prediction_ans in clean_reference_ans):
            judge = True
            # print(f"3 judge: {judge}")
        elif (len(clean_reference_ans) >= 3) and (not is_number(clean_reference_ans)) and (not clean_prediction_ans.startswith("-")) and (not clean_reference_ans.startswith("-")) and (clean_reference_ans in clean_prediction_ans):
            judge = True
            # print(f"4 judge: {judge}")
        else:
            judge = clean_prediction_ans == clean_reference_ans
            # print(f"5 judge: {judge}")
        if verbose:
            print(f"clean_prediction_ans: {clean_prediction_ans} | clean_reference_ans: {clean_reference_ans} | judge: {judge}")
        return judge, clean_prediction_ans, clean_reference_ans
    except Exception as e:
        print(e)
        return prediction_ans == reference_ans, prediction_ans, reference_ans


def is_college_correct(completion, answer, verbose=False):
    completion = completion.split("\n")[0].strip()
    completion = completion.lower()
    answer = answer.lower()

    # Extract short answer from completion
    extract_ans = None

    clean_reference_ans = strip_string(answer)
    is_reference_ans_number = is_number(clean_reference_ans)

    # First extract boxed answer
    # print(completion, answer)
    unbox_long_answer, box_short_answers = unbox_and_extract(completion)
    if box_short_answers != []:
        extract_ans = box_short_answers[-1].strip()
        # print(f"1 extract_ans: {extract_ans}")
    # extract the last number answer
    elif is_reference_ans_number:
        numbers = re.findall(r"[\-+]?\d*[\.,/]?\d+", completion)
        if numbers: 
            extract_ans = numbers[-1]
        # print(f"2 extract_ans: {extract_ans}")
    # extract "the answer is ..." answer
    elif ("answer is" in completion) or ("solution is" in completion):
        if "answer is" in completion:
            split_ans = completion.split('answer is')
        else:
            split_ans = completion.split('solution is')
        ans = split_ans[-1].strip().lstrip(":").strip()
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        extract_ans_temp = extract_ans_temp.strip('.')
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        # print(f"3 extract_ans: {extract_ans}")
    # extract "therefore xx is xxx" answer
    elif "is" in completion:
        pos = completion.rfind("is")
        ans = completion[pos+2:].strip().lstrip(":").strip()
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        extract_ans_temp = extract_ans_temp.strip('.')
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        # print(f"4 extract_ans: {extract_ans}")
    else:
        # print(completion)
        extract_ans = strip_string(completion)
        # return False, f"failed extracting answer from completion", clean_reference_ans

    judge, clean_prediction_ans, clean_reference_ans = is_equiv(extract_ans, answer, verbose=verbose)
    return judge, clean_prediction_ans, clean_reference_ans


"""
Evaluate math problems: TAL-SCQ
"""
import re
import sympy
from sympy.parsing.latex import parse_latex

SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), ('\%', '%'),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer

def check_sympy_equivalence(formatted_target_str, formatted_prediction_str):
    flag = False    
    try:
        target_expr = parse_latex(formatted_target_str)
    except:
        target_expr = formatted_target_str
        flag = True
    
    try:
        prediction_expr = parse_latex(formatted_prediction_str)
    except:
        prediction_expr = formatted_prediction_str
        flag = True
    
    if flag == True:
        return formatted_target_str == formatted_prediction_str

    try:
        return sympy.simplify(target_expr - prediction_expr) == 0
    except:
        return False

def is_talscq_correct(pred_str, ans_str):
    # import pdb; pdb.set_trace()
    # pattern = "#### (.*)$"

    # if "Question" in pred_str:
    #     pred_str = pred_str.split("Question")[0]

    # preds = re.findall(pattern, pred_str)
    # pred = preds[-1] if len(preds) >= 1 else ""
    # if "</s>" in pred:
    #     pred = pred[:-4]
    
    pred_str = pred_str.split("\n")[0].strip()
    # gold = ans_str.replace("$", "")
    pred = normalize_final_answer(pred_str).lower()
    gold = normalize_final_answer(ans_str).lower()
    return check_sympy_equivalence(gold, pred), pred, gold


"""
Evaluate math problems: TheoremQA
"""

# Copyright from math/modeling/dataset/util.py recipe: https://github.com/TIGER-AI-Lab/TheoremQA/blob/main/number_utils.py

import re
import math
from latex2sympy2 import latex2sympy
from math import sqrt, sin, cos, log, pi, factorial, exp, e
E = 2.718


def floatify(num: str):
    try:
        num = float(num)
        if num.is_integer():
            return round(num)
        else:
            return num
    except Exception:
        return None


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def clean_units(pred_str: str):
    """Clean the units in the number."""
    def convert_pi_to_number(code_string):
        code_string = code_string.replace('\\pi', 'π')
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r'(?<![\d}])\\?π', '3.14', code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r'(\d)(\\?π)', r'\1*3.14', code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r'\{(\\?π)\}', '3.14', code_string)
        code_string = re.sub(r'\*(\\?π)', '*3.14', code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace('%', '/100')
    pred_str = pred_str.replace('$', '')
    pred_str = pred_str.replace('¥', '')
    pred_str = pred_str.replace('°C', '')
    pred_str = pred_str.replace(' C', '')
    pred_str = pred_str.replace('°', '')
    return pred_str


def number_it(num):
    if isinstance(num, (int, float)):
        return num

    num = clean_units(num)
    try:
        num = str(latex2sympy(num))
    except Exception:
        pass

    if floatify(num) is not None:
        return floatify(num)
    else:
        try:
            num = eval(num)
            if isinstance(num, list) or isinstance(num, tuple):
                num = num[0]
            if floatify(num) is not None:
                return floatify(num)
            else:
                return None
        except Exception:
            return None


def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False
        if isinstance(gt, int):
            return round(p) == gt
        else:
            return within_eps(pred=p, gt=gt)
    except Exception:
        return False


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    elif len(pred) != len(gt):
        return False
    elif any([not isinstance(x, (int, float)) for x in pred]):
        return False
    else:
        pred = sorted(pred)
        gt = sorted(gt)
        return all([compare_two_numbers(p, g) for p, g in zip(pred, gt)])


# Copyright from math/modeling/dataset/util.py recipe: https://github.com/TIGER-AI-Lab/TheoremQA/blob/main/utils.py
def extract_theoremqa_answer(pred: str, answer_flag: bool = True):
    if any([option in pred.lower() for option in ['yes', 'true']]):
        pred = 'True'
    elif any([option in pred.lower() for option in ['no', 'false']]):
        pred = 'False'
    elif any([option in pred.lower() for option in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']]):
        pass
    else:
        if answer_flag:
            # Extract the numbers out of the string
            pred = pred.split('=')[-1].strip()
            pred = clean_units(pred)
            try:
                tmp = str(latex2sympy(pred))
                pred = str(eval(tmp))
            except Exception:
                if re.match(r'-?[\d\.]+\s\D+$', pred):
                    pred = pred.split(' ')[0]
                elif re.match(r'-?[\d\.]+\s[^\s]+$', pred):
                    pred = pred.split(' ')[0]
        else:
            # desparate search over the last number
            preds = re.findall(r'-?\d*\.?\d+', pred)
            if(len(preds) >= 1):
                pred = preds[-1]
            else:
                pred = ''

    return pred


def answer_clean(direct_answer_trigger_for_fewshot: tuple, pred: str):
    pred = pred.strip('\n')

    # Determine if this is ICL, if so, use \n\n to split the first chunk.
    ICL = False
    for trigger in direct_answer_trigger_for_fewshot:
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split('\n\n')[0]

    # Split the trigger to find the answer.
    preds = re.split('|'.join(direct_answer_trigger_for_fewshot), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip('\n').rstrip('.').rstrip('/').strip(' ')

    pred = [extract_theoremqa_answer(pred, answer_flag)]

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            # choose the first element in list ...
            pred = pred[0]
        else:
            # choose the last e
            pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip('.').rstrip('/')

    return pred


def compare_answer_with_groundtruth(answer: str, groundtruth_str: str, groundtruth_num = None):
    if groundtruth_str.lower() in ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']:
        return groundtruth_str.lower() in answer.lower()
    elif answer.lower() == groundtruth_str.lower():
        return True
    elif groundtruth_num is not None:
        if isinstance(groundtruth_num, (int, float)):
            return compare_two_numbers(number_it(answer), groundtruth_num)
        else:
            if answer.startswith('(') and answer.endswith(')'):
                try:
                    answer = list(eval(answer))
                    answer = [number_it(a) for a in answer]
                except Exception as e:
                    return False
                return compare_two_list(answer, groundtruth_num)
            else:
                return False
    else:
        return False


def is_theorem_correct(output_str, answer_str):
    output_str = output_str.split('\n')[0].strip().lower()
    answer_str = answer_str.lower()
    # import pdb; pdb.set_trace()
    equiv = compare_answer_with_groundtruth(output_str, answer_str)

    return equiv, answer_str, answer_str


"""
Evaluate math problems: MATH
"""

# Copyright from math/modeling/dataset/util.py recipe: https://github.com/hendrycks/math/blob/main/modeling/dataset/util.py
import pprint

def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]

def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string


# Copyright from math/modeling/math_equivalence.py recipe: https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv_math(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def is_math_correct(output_str, answer_str):
    output_str = last_boxed_only_string(output_str)
    answer_str = last_boxed_only_string(answer_str)

    output, answer = remove_boxed(output_str), remove_boxed(answer_str)

    # import pdb; pdb.set_trace()
    equiv = is_equiv_math(output, answer)

    return equiv, output, answer


"""
Evaluate math problems: GSM8K
"""

# Copyright from math/modeling/math_equivalence.py recipe: https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_gsm8k_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example)
    extract_output = extract_answer(model_completion)
    assert gt_answer != INVALID_ANS
    return int(extract_output == gt_answer), extract_output, gt_answer


"""
Evaluate part
"""
def compute_sample_score(label, output, dataset):
    # import pdb; pdb.set_trace()
    if dataset in ["triviaqa", "webqa"]:
        score = compute_exact(label, output)
        return score, output, label
    elif dataset in ["gsm8k"]:
        score, extract_output, extract_label = is_gsm8k_correct(output, label)
        return score, extract_output, extract_label
    elif dataset in ["math"]:
        score, extract_output, extract_label = is_math_correct(output, label)
        # assert extract_output in output and extract_label in label
        return score, extract_output, extract_label
    elif dataset in ["theorem"]:
        score, extract_output, extract_label = is_theorem_correct(output, label)
        return score, extract_output, extract_label
    elif dataset in ["talscq"]:
        score, extract_output, extract_label = is_talscq_correct(output, label)
        return score, extract_output, extract_label
    elif dataset in ["college"]:
        # import pdb; pdb.set_trace()
        score, extract_output, extract_label = is_college_correct(output, label)
        return score, extract_output, extract_label


def compute_scores(outputs, gold_answer=None, dataset=None):
    # For greedy decoding answers
    scores = []
    extract_outputs = []
    for output in outputs:
        score, extract_output, extract_answer = compute_sample_score(gold_answer, output, dataset)
        scores.append(score)
        extract_outputs.append(extract_output)
        average_score = sum(scores) / len(scores)

    return scores, average_score, extract_outputs, extract_answer


def evaluate():
    # Format output file.
    args.input_dir = os.path.join(args.input_dir.format(args.dataset), 
                                  "{}_{}_{}{}{}".format(args.model_name, 
                                                        args.dataset, 
                                                        args.model_suffix,
                                                        "_vllm" if args.vllm_use else "",
                                                        "_icl" if args.icl_use else ""))
    args.score_dir = os.path.join(args.score_dir.format(args.dataset), "{}_{}_{}_{}". \
                                   format(args.model_name, args.dataset, args.data_file, args.data_suffix))
    log_path = os.path.join(args.input_dir, "eval.log")
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Directory {args.input_dir} not found.")
    
    # print(log_path)
    # Set logging.
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )

    # Loading dataset.
    data_pool = []
    if args.split_num > 0:
        for split_id in range(args.split_num):
            data_path = os.path.join(args.input_dir, f"{args.data_file}_generate_{split_id}.json")
            logging.info(f"Load data from {data_path} ...")
            data_pool.extend(read_jsonl(data_path))
    else:
        data_path = os.path.join(args.input_dir, f"{args.data_file}_generate.json")
        logging.info(f"Load data from {data_path} ...")
        data_pool = read_json(data_path)

    output_path = os.path.join(args.input_dir, f"{args.data_file}_generate.json")    

    if args.score_use:
        score_path = os.path.join(f"{args.score_dir}.json")
        score_list = read_json(score_path)

        categories = {}
        categories_correct = {}
        for idx in range(len(Levels)):
            categories[Levels[idx+1]] = 0
            categories_correct[Levels[idx+1]] = 0
            
        for idx, data in enumerate(data_pool):
            assert data["question_id"] == score_list[idx]["question_id"]
            data_known = known_level(score_list[idx]["scores"]["greedy_scores_avg"], score_list[idx]["scores"]["sample_scores_avg"], args.dataset)
            categories[data_known] += 1
            score, extract_output, extract_answer = compute_sample_score(label=data["answer"], output=data["output"], dataset=args.dataset)
            # print(data_known, score)
            categories_correct[data_known] += score

            data["greedy_scores_avg"] = score_list[idx]["scores"]["greedy_scores_avg"]
            data["sample_scores_avg"] = score_list[idx]["scores"]["sample_scores_avg"]
            data["known_level"] = data_known

        assert sum(categories.values()) == len(data_pool)

        # logging.info(f"Categories: {categories}")
        # logging.info(f"Categories correct: {categories_correct}")
        logging.info(f"Total Accuracy: {sum(categories_correct.values())/sum(categories.values())}")
        for idx in range(len(Levels)):
            logging.info({
                "Level": Levels[idx+1],
                "Total": categories[Levels[idx+1]],
                "Correct": categories_correct[Levels[idx+1]],
                "Accuracy": round(categories_correct[Levels[idx+1]]/categories[Levels[idx+1]], 4)
            })
        
        write_json(output_path, data_pool)
    else:
        scores = 0
        for data in data_pool:
            # print("question_id: ", data["question_id"])
            score, norm_pred, norm_gold = compute_sample_score(label=data["answer"], output=data["output"], dataset=args.dataset)
            data["score"] = score
            data["extract_pred"] = norm_pred
            data["extract_gold"] = norm_gold
            scores += score

        logging.info(f"Total Accuracy: {scores/len(data_pool)}")
        write_json(output_path, data_pool)


if __name__=="__main__":
    evaluate()
