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

import os
import json
import random
import argparse

from utils import parallel_launcher, instantiate_client, get_criterion_fulfilment

def load_data(filename, debug):
    with open(filename, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    response_criterion_data = []

    for dp in data:
        response_field = "response"
        response = dp[response_field]
        domain = dp["domain"]
        task_id = dp["task_id"]
        prompt_tokens = dp["prompt_tokens"]
        completion_tokens = dp["completion_tokens"]
        
        for criterion_obj in dp["rubrics"]:
            criterion_description = criterion_obj["criterion_description"]
            criterion_type = criterion_obj["criterion_type"]
            criterion_weight = criterion_obj["criterion_weight"]
            response_criterion_data.append({"task_id": task_id, "domain":domain, "criterion_description": criterion_description, "criterion_weight": criterion_weight, "criterion_type": criterion_type, "response": response, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}) 

    # shuffling to increase the likelihood of cache hit rather than concurrent processing
    random.shuffle(response_criterion_data)

    if debug:
        response_criterion_data = response_criterion_data[:1]
    print("total data:", len(response_criterion_data))
    return response_criterion_data


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="run llm judge on generated reports")

    parser.add_argument('-f', "--filename", required=True)
    parser.add_argument('-ak', "--api-key", required=True)
    parser.add_argument('-m', "--model", default="openai/gpt-oss-120b", choices=["openai/gpt-oss-120b"])
    parser.add_argument('-l', '--library', default="openrouter", choices=["openrouter"])
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-p', "--parallel", type=int, default=32)
    parser.add_argument('-t', "--timeout", type=int, default=600)
    parser.add_argument('-ra', "--retry-attempts", type=int, default=3, help="retry attempts due to sproadic failures e.g. rate limiting or other issues")
    parser.add_argument('-of', "--output-folder", default="judgements_generated")

    args = parser.parse_args()
    reasoning = None # explicitly setting reasoning to None means reasoning effort varies based on criterion domain and type to high or low
    os.makedirs(args.output_folder, exist_ok=True)
    output_filename = os.path.join(args.output_folder, os.path.basename(args.filename))

    print(args)
    print(output_filename)
    response_criterion_data = load_data(args.filename, args.debug)

    client = instantiate_client(args.library, args.api_key, args.timeout)

    inference_hyperparameters = {"reasoning": reasoning}
    parallel_launcher(get_criterion_fulfilment, args.parallel, args.retry_attempts, response_criterion_data, output_filename, inference_hyperparameters, client, args.model)

