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

from sklearn.metrics import f1_score
import json
import numpy as np
import os
import argparse

correct_prefix = "Yes"

def get_metric(condition=None, value=None):
    y_pred = []
    y_true = []
    for dp in data:
        if condition is None or value in dp[condition]:
            human_annotation = dp["human_annotation"]
            y_true.append(int(human_annotation))
            judge_rating = dp["judge_rating"].strip()
            y_pred.append(int(judge_rating.startswith(correct_prefix)))

    return round(100*f1_score(y_true, y_pred, average='macro'),1)

def get_mean_pred_error(condition=None, value=None):
    y_pred = []
    y_true = []
    for dp in data:
        if condition is None or value in dp[condition]:
            human_annotation = dp["human_annotation"]
            y_true.append(int(human_annotation))
            judge_rating = dp["judge_rating"].strip()
            y_pred.append(int(judge_rating.startswith(correct_prefix)))

    y_diff = [i-j for i, j in zip(y_pred, y_true)]
    return round(100*np.mean(y_diff),1)

def get_tokens():
    prompt_tokens = [i["judge_prompt_tokens"] for i in data]
    completion_tokens = [i["judge_completion_tokens"] for i in data]
    return [round(np.mean(prompt_tokens)), round(np.mean(completion_tokens))]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="score llm judge")
    parser.add_argument('-f', "--base-filename", required=True)
    parser.add_argument('-if', "--input-folder", default="judgements")
    parser.add_argument('-of', "--output-folder", default="judgements_scores")
    args = parser.parse_args()

    input_filename = os.path.join(args.input_folder, args.base_filename)
    output_filename = os.path.join(args.output_folder, args.base_filename)
    os.makedirs(args.output_folder, exist_ok=True)

    with open(input_filename, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    
    fields = {
        "domain": ["Physics PhD", "Chemistry PhD", "Finance MBA", "Consulting MBA"],
        "criterion_type": ["Extraction (recall)", "Reasoning", "Style"]
    }

    results = {}
    for field in fields:
        for value in fields[field]:
            results[value] = get_metric(condition=field, value=value)

    results["All"] = get_metric()
    for model in ["o3", "r1-0528", "grok4"]:
        results[model] = get_mean_pred_error(condition="model", value=model)
    all_biases = [results[model] for model in ["o3", "r1-0528", "grok4"]]
    results["Bias-Index"] = round(max(all_biases) - min(all_biases), 3)
    results["MF1-BI"] = round(results["All"]-results["Bias-Index"], 3)
    results["prompt_tokens"] = get_tokens()[0]
    results["completion_tokens"] = get_tokens()[1]

    with open(output_filename, "w") as fw: 
        fw.write(json.dumps(results, indent=4))

