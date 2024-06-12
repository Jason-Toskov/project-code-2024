import argparse
import copy
import json
import multiprocessing
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, load_dataset
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
from trl import SFTTrainer


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
    tokenizer.padding_side = 'right' # to prevent errors with FA
    tokenizer.truncation_side = 'right' # to prevent cutting off last generation


    if "TIGER-Lab/MathInstruct" == input_args.data_path:
        dataset = load_dataset("TIGER-Lab/MathInstruct")
        system_msg = {"role": "system", "content": "You are an expert professor, teaching a student how to solve a problem by providing a full explanation of the solution."}
        
        text_samples = []
        for dp in tqdm(dataset["train"]):
            msg_qn = {"role": "user", "content": dp["instruction"]}
            msg_response = {"role": "assistant", "content": dp["output"]}
            if input_args.use_system_msg:
                txt = [system_msg, msg_qn, msg_response]
            else:
                txt = [msg_qn, msg_response]
            text_samples.append(txt)
            
        ds = Dataset.from_dict({"messages": text_samples})

        # def process(row):
        #     row["messages"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        #     return row

        # ds = ds.map(
        #     process,
        #     num_proc=multiprocessing.cpu_count(),
        #     load_from_cache_file=False,
        # )
        
        ds = ds.filter(lambda x: len(tokenizer.apply_chat_template(x["messages"], tokenize=True)) <= input_args.max_seq_length)
        
        ds = ds.train_test_split(test_size=0.05, seed=input_args.seed)
        train_dataset = ds["train"]
        eval_dataset = ds["test"]
        ds_size = len(dataset["train"])
    else:
        raise ValueError("Unknown dataset")
    
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
        warmup_ratio=0.05,                       # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",             # use cosine learning rate scheduler
        logging_steps=25,                       # log every 25 steps
        save_steps=1000,                         # when to save checkpoint
        save_total_limit=2,                     # limit the total amount of checkpoints
        evaluation_strategy="steps",            # evaluate every 1000 steps
        eval_steps=ds_size // input_args.batch_size // 10, # when to evaluate
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        push_to_hub=False,                      # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )
    
    trainer = SFTTrainer(
        model,
        peft_config=peft_config,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
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
    parser.add_argument("--run_name", type=str, default="llama3_sft_mathinstruct")
    parser.add_argument("--output_dir", type=str, default="/scratch/toskov/mnlp/output")
    parser.add_argument("--data_path", type=str, default="TIGER-Lab/MathInstruct")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--use_quantization", action="store_true")
    parser.add_argument("--use_system_msg", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=10107)
    parser.add_argument("--max_seq_length", type=int, default=704)
    
    
    args = parser.parse_args()
    main(args)
