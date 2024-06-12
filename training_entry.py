import argparse
import copy
import json
import multiprocessing
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Conversation,
    TrainingArguments,
    pipeline,
)
from trl import DPOTrainer


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main(input_args):
    set_seed(input_args.seed)
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation

    ds = Dataset.from_json(input_args.data_path)
    ds = ds.train_test_split(test_size=0.1, seed=input_args.seed)

    train_dataset = ds["train"]
    eval_dataset = ds["test"]
    
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    quantization_config = bnb_config if input_args.use_quantization else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        use_cache=False,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    
    args = TrainingArguments(
        output_dir=os.path.join(input_args.output_dir, input_args.run_name), # directory to save and repository id
        num_train_epochs=1,                     # number of training epochs
        per_device_train_batch_size=input_args.batch_size, #4        # batch size per device during training
        per_device_eval_batch_size=input_args.batch_size, #4           # batch size for evaluation
        gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",             # use cosine learning rate scheduler
        logging_steps=25,                       # log every 25 steps
        save_steps=500,                         # when to save checkpoint
        save_total_limit=2,                     # limit the total amount of checkpoints
        evaluation_strategy="steps",            # evaluate every 1000 steps
        eval_steps=700,                         # when to evaluate
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        push_to_hub=False,                      # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )
    
    dpo_args = {
        "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
        "loss_type": "sigmoid"                  # The loss type for DPO.
    }
    
    trainer = DPOTrainer(
        model,
        ref_model=None, # set to none since we use peft
        peft_config=peft_config,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=input_args.max_seq_length,
        max_prompt_length=input_args.prompt_length,
        beta=dpo_args["beta"],
        loss_type=dpo_args["loss_type"],
    )
    
    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()
    
    # save model at the end of training
    trainer.save_model()
    
    # save args
    with open(os.path.join(input_args.output_dir, input_args.run_name, "args.json"), "w") as f:
        json.dump(vars(input_args), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="llama3")
    parser.add_argument("--output_dir", type=str, default="/scratch/toskov/mnlp/output")
    parser.add_argument("--data_path", type=str, default="/scratch/toskov/mnlp/dpo_hf_dataset.json")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--use_quantization", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10107)
    parser.add_argument("--prompt_length", type=int, default=402)
    parser.add_argument("--max_seq_length", type=int, default=912)
    
    
    args = parser.parse_args()
    main(args)
