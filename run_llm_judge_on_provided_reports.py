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

from datasets import load_dataset

import random
import argparse
import os

from utils import parallel_launcher, get_criterion_fulfilment, instantiate_client

def load_data(debug):
    dataset = load_dataset("nvidia/ProfBench")["test"]
    response_criterion_data = []
    for dp in dataset:
        for model in ["o3", "grok4", "r1-0528"]:
            response_field = model + "_response"
            response = dp[response_field]
            domain = dp["domain"]
            task_id = dp["task_id"]
            for criterion_obj in dp["rubrics"]:
                criterion_description = criterion_obj["criterion_description"]
                criterion_type = criterion_obj["criterion_type"]
                criterion_weight = criterion_obj["criterion_weight"]
                human_annotation = criterion_obj[model+"_fulfilment"]
                response_criterion_data.append({"task_id": task_id, "domain":domain, "criterion_description": criterion_description, "criterion_weight": criterion_weight, "criterion_type": criterion_type, "model": model, "human_annotation": human_annotation, "response": response}) 

    # shuffling to increase the likelihood of cache hit of common prefix rather than concurrent processing
    random.shuffle(response_criterion_data)
    if debug:
        response_criterion_data = response_criterion_data[:1]
    print("total datapoints:", len(response_criterion_data))
    return response_criterion_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run llm judge on provided reports")

    parser.add_argument('-m', "--model", required=True)
    parser.add_argument('-ak', "--api-key", required=True)
    parser.add_argument('-l', '--library', choices=["openrouter", "openai"], default="openrouter")
    parser.add_argument('-r', '--reasoning', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-re', '--reasoning-effort', choices=["low", "medium", "high", "minimal"], default="high", help="this default to high because openrouter use this value to set budget_tokens for anthropic/gemini models")
    parser.add_argument('-p', "--parallel", type=int, default=32)
    parser.add_argument('-t', "--timeout", type=int, default=600)
    parser.add_argument('-ra', "--retry-attempts", type=int, default=3, help="retry attempts due to sproadic failures e.g. rate limiting or other issues")
    parser.add_argument('-f', "--folder", default="judgements")
    args = parser.parse_args()

    model = args.model

    reasoning_strength = args.reasoning if not args.reasoning else args.reasoning_effort

    os.makedirs(args.folder, exist_ok=True)

    clean_model_name = model.replace("/", "_").replace(":", "-")
    output_filename = f"{args.folder}/{clean_model_name}_reasoning_{reasoning_strength}.jsonl"

    response_criterion_data = load_data(args.debug)
    client = instantiate_client(args.library, args.api_key, args.timeout)

    inference_hyperparameters = {"reasoning": reasoning_strength}

    parallel_launcher(get_criterion_fulfilment, args.parallel, args.retry_attempts, response_criterion_data, output_filename, inference_hyperparameters, client, model)
