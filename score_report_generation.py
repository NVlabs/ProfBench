# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
from collections import defaultdict
import numpy as np
import argparse
import os



weight_to_scale = {"Critical": 4, "Major": 3, "Minor": 2, "Additional": 1}


def get_predicted_score_per_task_id_e2e(condition=None, value=None, data=None):
    task_id_response_to_max_score = defaultdict(int)
    task_id_response_to_achieved_score = defaultdict(int)
    task_id_to_domain =  defaultdict(str)
    criterion_type_to_fulfilment = defaultdict(list)

    for i in data:
        weight = i["criterion_weight"]
        scale = weight_to_scale[weight]
        task_id = i["task_id"]
        domain = i["domain"]
        criterion_types = i["criterion_type"]
        response = i["response"]

        task_id_response_to_max_score[(task_id, response)] += scale
        task_id_to_domain[task_id] = domain

        criterion_fulfilment = i[condition].startswith(value) if isinstance(value, str) else i[condition] == value
        
        if criterion_fulfilment: 
            task_id_response_to_achieved_score[(task_id, response)] += scale

        for criterion_type in criterion_types:
            criterion_type_to_fulfilment[criterion_type].append(int(criterion_fulfilment))
        
    task_id_to_scores = defaultdict(list)

    for task_id_response in task_id_response_to_max_score:
        task_id, response = task_id_response
        score = task_id_response_to_achieved_score[task_id_response] / task_id_response_to_max_score[task_id_response]
        task_id_to_scores[task_id].append(score)

    domain_to_scores = defaultdict(list)

    for task_id, adjusted_score in task_id_to_scores.items():
        domain = task_id_to_domain[task_id]
        domain_to_scores[domain].append(np.mean(adjusted_score))
    
    domain_average = {domain: round(np.mean(domain_to_scores[domain]),3) for domain in domain_to_scores}
    
    all_domains = round(np.mean(list(domain_average.values())), 3)

    domain_average['Overall'] = all_domains

    # add average of each criterion type to all_domains
    for criterion_type in criterion_type_to_fulfilment:
        domain_average[criterion_type] = round(np.mean(criterion_type_to_fulfilment[criterion_type]), 3)
    
    for key in domain_average:
        domain_average[key] = round(domain_average[key]*100, 1)
    
    prompt_tokens = list({i["task_id"]: i["prompt_tokens"] for i in data if isinstance(i["prompt_tokens"], int)}.values())
    completion_tokens = list({i["task_id"]:i["completion_tokens"] for i in data if isinstance(i["completion_tokens"], int)}.values())
    response_len = list({i["task_id"]:len(i["response"]) for i in data if i["response"]}.values())

    domain_average["prompt_tokens"] = round(np.mean(prompt_tokens))
    domain_average["completion_tokens"] = round(np.mean(completion_tokens))
    domain_average["response_len_chars"] = round(np.mean(response_len))
    return domain_average

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="score report generation")
    parser.add_argument('-f', "--base-filename", required=True)
    parser.add_argument('-if', "--input-folder", default="judgements_generated")
    parser.add_argument('-of', "--output-folder", default="scores")
    args = parser.parse_args()

    input_filename = os.path.join(args.input_folder, args.base_filename)
    output_filename = os.path.join(args.output_folder, args.base_filename)
    os.makedirs(args.output_folder, exist_ok=True)

    with open(input_filename, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    results = get_predicted_score_per_task_id_e2e(condition="judge_rating", value="Yes", data=data)

    with open(output_filename, "w") as fw: 
        fw.write(json.dumps(results, indent=4))