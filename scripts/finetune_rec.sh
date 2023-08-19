#!/bin/bash
##Original Llama
exp_tag="llama"
python finetune_rec.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'data/beauty_sequential_single_prompt_train_sample.json' \
    --valid_data_path 'data/beauty_sequential_single_prompt_test.json' \
    --output_dir './lora-LLMRec-'$exp_tag \
    --prompt_template_name 'rec_template' \
    --micro_batch_size 2 \
    --batch_size 32 \
    --cutoff_len 512 \
    --save_eval_steps 5 \
    --wandb_run_name $exp_tag
##Original alpaca
exp_tag="alpaca"
python finetune_rec.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'data/beauty_sequential_single_prompt_train_sample.json' \
    --valid_data_path 'data/beauty_sequential_single_prompt_test.json' \
    --output_dir './lora-LLMRec-'$exp_tag \
    --prompt_template_name 'rec_template' \
    --micro_batch_size 2 \
    --batch_size 32 \
    --cutoff_len 512 \
    --save_eval_steps 5 \
    --resume_from_checkpoint 'alpaca-lora-7b' \
    --lora_r 16 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]'\
    --wandb_run_name $exp_tag