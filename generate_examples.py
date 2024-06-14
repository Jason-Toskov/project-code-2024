import copy
import json
import multiprocessing

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm.notebook import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Conversation,
    TrainingArguments,
    pipeline,
)
from trl import DPOTrainer

ds = Dataset.from_json("/scratch/toskov/mnlp/dpo_hf_dataset_nosysmsg.json")
ds = ds.train_test_split(test_size=0.1, seed=10107)

train_dataset = ds["train"]
eval_dataset = ds["test"]

# Path to saved peft adapter model
peft_model_id = "/scratch/toskov/mnlp/output/llama3_dpo_mathinstruct_base"

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  device_map="auto",
  torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

# load into pipeline
pipe = pipeline("conversational", model=model, tokenizer=tokenizer)

messages = []
for i in range(100):
    conversation = Conversation()
    # conversation.add_message(system_msg)
    conversation.add_message({"role": "user", "content": eval_dataset[i]["prompt"].split("<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]})

    conversation = pipe(conversation, max_new_tokens=2048, do_sample=True, temperature=1.0, top_k=50, top_p=0.9, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

    conversation.add_message({"role": "user", "content": "Now, based off of the explanation you have given, give the correct answer as a single letter only, e.g. `A`, `B`, `C` or `D`."})

    conversation_answer = pipe(copy.deepcopy(conversation), max_new_tokens=16, do_sample=True, temperature=1.0, top_k=50, top_p=0.9, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    conversation_answer.messages[-1]["content"]
    
    messages.append(copy.deepcopy(conversation_answer.messages))
    
    with open("llama3_dpo_mathinstruct_base_nosysmsg_gens.json", "w") as f:
        json.dump(messages, f)