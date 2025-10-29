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
import copy
import base64
import argparse
import random

from datasets import load_dataset

from utils import parallel_launcher, instantiate_client

try:
    from google.genai.types import GenerateContentConfig, Tool, GoogleSearch, ThinkingConfig, Part
except:
    print("Google GenAI failed to import; ignore if not using otherwise please pip install google-genai")

def load_hf_data():
    dataset = load_dataset("nvidia/ProfBench")["test"]
    processed_data = []
    for dp in dataset:
        domain = dp["domain"]
        task_id = dp["task_id"]
        prompt = dp["prompt"]
        filepaths = dp["filepaths"] if "filepaths" in dp else []
        processed_data.append({"task_id": task_id, "domain":domain, "prompt": prompt, "filepaths":filepaths, "rubrics": dp["rubrics"]})
    return processed_data

def filter_data(prompt_data, version):
    if version == "debug":
        prompt_data = prompt_data[:1]
    elif version == "full":
        prompt_data = prompt_data * 16
    elif version == "lite":
        task_id_to_samples = {'Chem-0': 4, 'Chem-1': 4, 'Fin-0': 4, 'Fin-1': 4, 'Cons-0': 4, 'Fin-2': 4, 'Phys-0': 4, 'Phys-1': 4, 'Phys-2': 4, 'Fin-3': 4, 'Chem-2': 4, 'Chem-3': 5, 'Chem-4': 3, 'Phys-3': 4, 'Cons-1': 3, 'Phys-4': 4, 'Fin-4': 4, 'Cons-2': 3, 'Fin-5': 4, 'Cons-3': 4, 'Chem-5': 4, 'Fin-6': 4, 'Chem-6': 5, 'Chem-7': 4, 'Chem-8': 4, 'Phys-5': 4, 'Cons-4': 4, 'Phys-6': 4, 'Chem-9': 5, 'Phys-7': 4, 'Fin-7': 4, 'Cons-5': 3, 'Phys-8': 3, 'Cons-6': 4, 'Fin-8': 5, 'Fin-9': 5, 'Cons-7': 3, 'Cons-8': 4, 'Cons-9': 5, 'Phys-9': 4}
        new_data = []
        for dp in prompt_data:
            task_id = dp["task_id"]
            n_samples = task_id_to_samples[task_id]
            for i in range(n_samples):
                new_data.append(copy.deepcopy(dp))
        # shuffling to increase the likelihood of cache hit of common prefix rather than concurrent processing
        random.shuffle(new_data)
        prompt_data = new_data
    print("total samples:", len(prompt_data))
    return prompt_data

def get_openai_existing_filename_to_file_id():
    existing_file_list = client.files.list()
    existing_filename_to_file_id = {file_obj.filename: file_obj.id for file_obj in existing_file_list.data}
    return existing_filename_to_file_id

def get_openai_response(prompt, client=None, model=None, filepaths=None, reasoning=False, upload_documents=False, web_search=False):
    if upload_documents:
        existing_filename_to_file_id = get_openai_existing_filename_to_file_id()
        filepath_to_file_id = {}
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            if filename in existing_filename_to_file_id:
                filepath_to_file_id[filepath] = existing_filename_to_file_id[filename]
            else:
                file = client.files.create(
                    file=open(filepath, "rb"),
                    purpose="user_data"
                )
                filepath_to_file_id[filepath] = file.id

        content = [{"type": "input_file", "file_id": file_id} for _, file_id in filepath_to_file_id.items()]
        content += [{"type": "input_text", "text": prompt}]
    else:
        content = [{"type": "input_text", "text": prompt}]


    if web_search:
        tools = [{"type": "web_search_preview"}]
    else:
        tools = []

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=64000 if reasoning else 32000,
        tools=tools,
        reasoning={"effort": reasoning} if reasoning else {}
    )
    return response.output_text, response.usage.input_tokens, response.usage.output_tokens

def get_google_response(prompt, filepaths=None, client=None, model=None, reasoning=False, upload_documents=False, web_search=False):
    if upload_documents:
        pdf_parts = []
        for path in filepaths:
            with open(path, "rb") as f:
                pdf_parts.append(Part.from_bytes(data=f.read(), mime_type="application/pdf"))
        contents = pdf_parts + [prompt]
    else:
        contents = prompt

    if web_search:
        tools = [Tool(google_search=GoogleSearch())]
    else:
        tools = []
    
    max_tokens = 64000 if reasoning else 32000

    if not reasoning:
        thinking_budget = 0
    elif reasoning == "low":
        thinking_budget = int(0.2*max_tokens)
    elif reasoning == "medium":
        thinking_budget = int(0.5*max_tokens)
    elif thinking_budget == "high":
        thinking_budget = int(0.8*max_tokens)

    response = client.models.generate_content(model=model, contents=contents, config=GenerateContentConfig(tools=tools, thinking_config=ThinkingConfig(thinking_budget=thinking_budget)))
    
    thinking_tokens = response.usage_metadata.thoughts_token_count if response.usage_metadata.thoughts_token_count is not None else 0
    output_tokens = thinking_tokens + response.usage_metadata.candidates_token_count
    return response.text, response.usage_metadata.prompt_token_count, output_tokens

def encode_pdf_to_base64(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')
            
def get_openrouter_response(prompt, client=None, model=None, filepaths=None, reasoning=False, upload_documents=False, web_search=False):
    if upload_documents:
        prompt_content = [{
            "type": "file",
            "file": {
                "filename": "document.pdf",
                "file_data": f"data:application/pdf;base64,{encode_pdf_to_base64(pdf_path)}"
            }
        } for pdf_path in filepaths]
        prompt_content += [{ "type": "text", "text": prompt}]

    else:
        prompt_content = prompt

    messages = [{"role": "user", "content": prompt_content}]

    request_obj = {
        "model": model+":online" if web_search else model, # this uses Exa AI plugin
        "messages": messages,
        "temperature": 0.6 if reasoning else 0,
        "top_p": 0.95 if reasoning else 0,
        "max_tokens": 64000 if reasoning else 32000,
        "stream": False
    }

    if isinstance(reasoning, str):
        request_obj["reasoning_effort"] = reasoning

    completion = client.chat.completions.create(**request_obj)
    return completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens

def get_model_response(dp, idx, inference_hyperparameters, client, model):

    if client.library == "openrouter":
        func = get_openrouter_response
    elif client.library == "openai":
        func = get_openai_response
    elif client.library == "google":
        func = get_google_response

    response, prompt_tokens, completion_tokens = func(dp["prompt"], filepaths=dp["filepaths"], client=client, model=model, **inference_hyperparameters)
    
    dp["idx"] = idx
    dp["response"] = response
    dp["prompt_tokens"] = prompt_tokens
    dp["completion_tokens"] = completion_tokens
    return dp

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run report generation")

    parser.add_argument('-m', "--model", required=True)
    parser.add_argument('-ak', "--api-key", required=True, help="used as project for google vertexai")
    parser.add_argument('-v', '--version', choices=["debug", "lite", "full"], default="lite")
    parser.add_argument('-l', '--library', choices=["openrouter", "openai", "google"], default="openrouter", help="please use openrouter to use chat.completions interface and openai to use responses API")
    parser.add_argument('-bu', '--base-url', default=None, help="if set, it will instantiate an openai client with this base_url instead of the default for each library")
    parser.add_argument('-ws', '--web-search', action='store_true')
    parser.add_argument('-r', '--reasoning', action='store_true')
    parser.add_argument('-re', '--reasoning-effort', choices=["low", "medium", "high", "minimal"], default="high", help="this default to high because openrouter use this value to set budget_tokens for anthropic/gemini models")
    parser.add_argument('-p', "--parallel", type=int, default=32)
    parser.add_argument('-t', "--timeout", type=int, default=600)
    parser.add_argument('-ra', "--retry-attempts", type=int, default=3, help="retry attempts due to sproadic failures e.g. rate limiting or other issues")
    parser.add_argument('-f', "--folder", default="inference")
    args = parser.parse_args()

    model = args.model

    reasoning = args.reasoning if not args.reasoning else args.reasoning_effort
    upload_documents = 0 # disable document upload in public release
    web_search = int(args.web_search)

    os.makedirs(args.folder, exist_ok=True)
    clean_judge_name = model.replace("/", "_").replace(":", "-")
    output_filename = f"{args.folder}/{clean_judge_name}_reasoning_{reasoning}_search_{web_search}.jsonl"

    prompt_data = load_hf_data()
    prompt_data = filter_data(prompt_data, args.version)

    client = instantiate_client(args.library, args.api_key, args.timeout, base_url=args.base_url)

    inference_hyperparameters = {
        "reasoning": reasoning,
        "upload_documents": upload_documents,
        "web_search": web_search
    }
    parallel_launcher(get_model_response, args.parallel, args.retry_attempts, prompt_data, output_filename, inference_hyperparameters, client, model)
