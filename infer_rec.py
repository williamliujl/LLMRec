import os
os.environ['TRANSFORMERS_CACHE'] = '/data/private/peilin/cache'
os.environ['HF_HOME'] = '/data/private/peilin/cache'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# export HF_HOME=/path/to/cache/directory
import sys
import json

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
import time
import pandas as pd

from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"

def load_instruction(instruct_dir):
    input_data = []
    with open(instruct_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            d = json.loads(line)
            input_data.append(d)
    return input_data

def load_instruction_from_csv(instruct_dir, prompt_idx='all'):
    input_data = []
    df = pd.read_csv(instruct_dir, dtype='str',usecols=['prompt','target','task','few_zero'])#todo:for test,you need to remove nrows
    dict_from_df = df.to_dict(orient='index')
    for key,value in dict_from_df.items():
        data = {}
        data['output'] = value['target'].strip()
        data['instruction'] = value['prompt'].strip()
        input_data.append(data)
    # print(input_data)
    return input_data, df

def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    # cache_dir: str = "/data/private/peilin/cache/huggingface",
    # the infer data, if not exists, infer the default instructions in code
    instruct_dir: str = "./data/beauty_sequential_single_prompt_test_sample.json",
    # output_type: str = "option",
    output_dir: str = "output/",
    use_lora: bool = False,
    lora_weights: str = "tloen/alpaca-lora-7b",
    # The prompt template to use, will default to med_template.
    prompt_template: str = "rec_template",
    max_new_tokens: int = 10,
    num_return_sequences: int = 10,
    num_beams: int = 10,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_lora:
        print(f"using lora {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
            )
        # if num_return_sequences > 1:
        answer_list = []
        for i in range(num_return_sequences):
            s = generation_output.sequences[i]
            output = tokenizer.decode(s)
            answer_list.append(prompter.get_response(output))
        return '\t'.join(answer_list)
    def write_to_json(data, output_dir):
        with open(output_dir, 'w') as f:
            for item in data:
                json.dump(item, f)
                f.write('\n')

    def infer_from_json(instruct_dir):
        # start = time.time()
        input_data = load_instruction(instruct_dir)
        output_data = []
        # i = 0
        for d in tqdm(input_data):
            # i+=1
            # if i == 10:
            #     break
            instruction = d["source"]
            output = d["target"]
            print("###infering###")
            model_output = evaluate(instruction)
            print("###instruction###")
            print(instruction)
            print("###golden output###")
            print(output)
            print("###model output###")
            print(model_output)
            task_type = d["task_type"]
            output_data.append({'labels':d['target'],'predict':model_output})
        file_name = "rec_{}_{}.json".format(task_type, base_model.split('/')[-1]) if not use_lora else 'rec_{}_{}_{}.json'.format(task_type, base_model.split('/')[-1], lora_weights.split('/')[-1])
        output_path = output_dir + file_name
        write_to_json(output_data, output_path)

    if instruct_dir != "":
        filename, file_extension = os.path.splitext(instruct_dir)
        file_extension_without_dot = file_extension[1:]
        if file_extension_without_dot == 'json':
            infer_from_json(instruct_dir)
        else:
            raise ValueError
    else:
        raise ValueError

if __name__ == "__main__":
    fire.Fire(main)
