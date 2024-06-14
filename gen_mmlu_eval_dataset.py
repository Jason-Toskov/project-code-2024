from datasets import load_dataset
from tqdm import tqdm
from utils import write_jsonl

d_compsci = load_dataset("cais/mmlu", "college_computer_science")
d_ml = load_dataset("cais/mmlu", "machine_learning")

mmlu = []
letters = ["A", "B", "C", "D"]
for dataset in [d_compsci["test"], d_ml["test"]]:
    for dp in tqdm(dataset):
        new_entry = {}
        new_entry["subject"] = dp["subject"]
        
        new_entry["question"] = "Question: " + dp["question"] + "\n\nOptions:\n"
        for letter, answer in zip(letters, dp["choices"]):
            new_entry["question"] += f"{letter}. {answer}\n"
        new_entry["question"] += "\nAnswer:"
        
        new_entry["answer"] = letters[dp["answer"]]
        
        mmlu.append(new_entry)
    
write_jsonl(mmlu, "mmlu_ml_compsci_eval_set.jsonl")
