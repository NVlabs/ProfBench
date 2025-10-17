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

import concurrent.futures
import json
from tqdm import tqdm

from openai import OpenAI


def instantiate_client(library, api_key, timeout, base_url=None):
    if base_url is not None:
        client = OpenAI(api_key=api_key, timeout=timeout, base_url=base_url)
    elif library == "openai":
        client = OpenAI(api_key=api_key, timeout=timeout)
    elif library == "openrouter":
        client = OpenAI(api_key=api_key, timeout=timeout, base_url="https://openrouter.ai/api/v1")
    elif library == "google":
        from google import genai
        # only tested on vertexai as we only have access to vertexai endpoint
        client = genai.Client(vertexai=True, project=api_key, location="global")
    client.library = library # set as marker for some inference func to use
    return client


def parallel_launcher(func, parallel, retry_attempts, data, output_filename, inference_hyperparameters, client, model):
    for j in range(retry_attempts):
        with open(output_filename, "a+") as fw:
            fw.seek(0)
            existing_data = [json.loads(line) for line in fw.readlines()]
            existing_indices = set([i['idx'] for i in existing_data])
            print("existing data:", len(existing_data))
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [executor.submit(func, dp, idx, inference_hyperparameters, client, model) for idx, dp in enumerate(data) if idx not in existing_indices]
                for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc=f"Attempt {j+1}"
                    ):
                    try:
                        full_result = future.result()
                        fw.write(json.dumps(full_result) + '\n')
                    except Exception as e:
                        print("Error from worker:", e)

def get_llm_judge_response(client, model, messages, reasoning_effort=None):
    request_obj = {
        "model": model,
        "messages": messages,
        "temperature": 0.6 if reasoning_effort else 0,
        "top_p": 0.95 if reasoning_effort else 0,
        "max_tokens": 32768 if reasoning_effort else 1,
        "stream": False
    }

    # to control the reasoning_effort
    if isinstance(reasoning_effort, str): 
        request_obj["reasoning_effort"] = reasoning_effort

    completion = client.chat.completions.create(
        **request_obj
    )
    return completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens

QUESTION_PROMPT_YES_NO = "Response:\n\n{response}\n\nEvaluate whether the response above satisfies this criterion: {criterion_description}. Only answer Yes or No."

def get_criterion_fulfilment(dp, idx, inference_hyperparameters, client, model):
    prompt = QUESTION_PROMPT_YES_NO.format(response=dp["response"], criterion_description=dp["criterion_description"])
    messages = [{"role": "user", "content": prompt}]

    reasoning_strength = inference_hyperparameters["reasoning"]

    # set reasoning effort to varying if explicitly set to None
    if reasoning_strength is None:
        reasoning_strength = "high" if (dp['domain'] in ["Physics PhD", "Chemistry PhD"] or 'Style' in dp["criterion_type"]) else "low"

    rating, prompt_tokens, completion_tokens = get_llm_judge_response(client, model, messages, reasoning_effort=reasoning_strength)
    dp["judge_rating"] = rating
    dp["judge_prompt_tokens"] = prompt_tokens
    dp["judge_completion_tokens"] = completion_tokens
    dp["idx"] = idx
    return dp