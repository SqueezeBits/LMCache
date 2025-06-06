import argparse
import json
import os
from typing import List

import numpy as np
from transformers import AutoTokenizer
import pandas as pd

parser = argparse.ArgumentParser(description="Process data percentage.")
parser.add_argument(
    "--parse",
    type=float,
    default=1,
    help="The percentage of data to process (0 to 1). Default is 1 (100%).")
parser.add_argument(
    "--human",
    type=List[str],
    default=["human", "user"],
    help="The list of human ids to use.")
parser.add_argument(
    "--gpt",
    type=List[str],
    default=["chatgpt", "bing", "bard", "gpt"],
    help="The list of gpt ids to use.")
parser.add_argument(
    "--system",
    type=List[str],
    default=["system"],
    help="The list of system ids to use.")

args = parser.parse_args()

with open('ShareGPT_V3_unfiltered_cleaned_split.json', 'r',
          encoding='utf-8') as file:
    data = json.load(file)

froms = set()
for d in data:
    for conv in d['conversations']:
        froms.add(conv['from'])
if froms != set(args.human + args.gpt + args.system):
    raise ValueError(
        f"Unknown ids found in the data: {froms - set(args.human + args.gpt + args.system)}. Please specify the correct ids in the command line."
    )

def estimate_num_tokens(text: str) -> int:
    if not hasattr(estimate_num_tokens, "tokenizer"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        estimate_num_tokens.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-14B")
    return len(estimate_num_tokens.tokenizer.tokenize(text))


num_of_ids = len(data)
print(f"Number of IDs: {num_of_ids}")
data = data[:int(num_of_ids * args.parse)]

count = 0

idx_to_remove = []
for idx, d in enumerate(data):
    if len(d['conversations']) > 1:
        current_human = d['conversations'][0]['from'] in args.human
        for conv in d['conversations'][1:]:
            if conv['from'] in args.human:
                if current_human:
                    idx_to_remove.append(idx)
                    break
                current_human = True
            elif conv['from'] in args.gpt:
                if not current_human:
                    idx_to_remove.append(idx)
                    break
                current_human = False

print(f"Removing {len(idx_to_remove)} data due to consecutive human or gpt rounds.")
data = [d for idx, d in enumerate(data) if idx not in idx_to_remove]

# Data collection for plots
all_num_rounds = []
all_prompt_lengths = []
all_output_lengths = []
all_prompt_lengths_with_history = []
plot_data_for_df = []

for d in data:
    if len(d['conversations']) > 0 and d['conversations'][0]['from'] == 'system':
        del d['conversations'][0]  # Remove system prompt
    d['num_round'] = len(
        d['conversations'])  # human is one round, gpt is another round
    human_tokens = []
    gpt_tokens = []
    cumulative_tokens = 0
    for conv in d['conversations']:
        num_tokens = estimate_num_tokens(conv['value'])
        if conv['from'] in args.human:
            human_tokens.append(num_tokens)
        elif conv['from'] in args.gpt:
            all_prompt_lengths_with_history.append(cumulative_tokens)
            conv['num_tokens'] = num_tokens
            gpt_tokens.append(num_tokens)
        cumulative_tokens += num_tokens

    all_num_rounds.append(d['num_round'])
    all_prompt_lengths.extend(human_tokens)
    all_output_lengths.extend(gpt_tokens)
    if human_tokens:
        plot_data_for_df.append({
            'num_round': d['num_round'],
            'total_prompt_length': sum(human_tokens),
            'total_output_length': sum(gpt_tokens),
            'last_prompt_length': human_tokens[-1] if human_tokens else 0,
            'last_history_length': sum(human_tokens[:-1]) if len(human_tokens) > 1 else 0 + sum(gpt_tokens[:-1]) if len(gpt_tokens) > 1 else 0,
            'last_output_length': gpt_tokens[-1] if gpt_tokens else 0,
        })

    if len(human_tokens) == 0:
        d['average_human_token'] = 0
        d['max_human_token'] = 0
    else:
        d['average_human_token'] = float(np.mean(human_tokens))
        d['max_human_token'] = float(np.max(human_tokens))
    if len(gpt_tokens) == 0:
        d['average_gpt_token'] = 0
        d['max_gpt_token'] = 0
    else:
        d['average_gpt_token'] = float(np.mean(gpt_tokens))
        d['max_gpt_token'] = float(np.max(gpt_tokens))

    count += 1
    print(f"Finished {count}")

with open('ShareGPT.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

with open('ShareGPT_stats.json', 'w', encoding='utf-8') as file:
    json.dump({
        'num_rounds': all_num_rounds,
        'prompt_lengths': all_prompt_lengths,
        'output_lengths': all_output_lengths,
        'prompt_lengths_with_history': all_prompt_lengths_with_history,
        'plot_data_for_df': plot_data_for_df,
    }, file, ensure_ascii=False, indent=2)
