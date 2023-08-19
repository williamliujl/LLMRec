#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python infer_rec.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './lora-LLMRec-llama' \
    --use_lora True \
    --instruct_dir './data/beauty_sequential_single_prompt_test.json' \
    --prompt_template 'rec_template' \
    --max_new_tokens 256 \
    --num_return_sequences 10 \
    --num_beams 10
CUDA_VISIBLE_DEVICES=0 python infer_rec.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './lora-LLMRec-alpaca' \
    --use_lora True \
    --instruct_dir './data/beauty_sequential_single_prompt_test.json' \
    --prompt_template 'rec_template' \
    --max_new_tokens 256 \
    --num_return_sequences 10 \
    --num_beams 10