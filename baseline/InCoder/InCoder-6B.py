import pickle
import os
import json
import datasets
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

input_dir = '../../datasets/MBPP/mbpp_test.csv'
input_column = 'prompt_testcase' #'prompt_testcase_llm'
output_filename = 'save/incoder-6B_mbpp_testcase_beam'
batch_size = 1

checkpoint = "facebook/incoder-6B"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = "<pad>"
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'left'
model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                             revision="float16",
                                             torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True,
                                             device_map="auto",
                                            #  load_in_8bit=True,
                                            ) 
print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")

# Convert the tokenized inputs into a PyTorch dataset
test_df = pd.read_csv(input_dir)
test_inputs_text = list(test_df[input_column])
test_inputs = tokenizer(test_inputs_text, padding=True, truncation=True)
test_dataset = datasets.Dataset.from_dict({
    "input_ids": test_inputs["input_ids"],
    "attention_mask": test_inputs["attention_mask"],
})
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))

generation_kwargs = {
    "min_new_tokens": 5,
    "max_new_tokens": 300,
    "pad_token_id": tokenizer.bos_token_id, #<|endoftext|>
    "eos_token_id": tokenizer.bos_token_id,
    "early_stopping": True,
    "num_return_sequences": 4,
    "num_beams": 4
}

predictions = []
curr_idx = 0
skip = False
if os.path.isfile(f'{output_filename}.json'):
    skip = True
    rfile = open(f"{output_filename}.json", "r")
    curr_idx = len(rfile.readlines())
    rfile.close()
file = open(f'{output_filename}.json', "a") 
for i, batch in enumerate(tqdm(test_loader)):
    if i < curr_idx:
        continue
    if skip:
        print(f'continue at idx: {i}')
        skip = False
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], clean_up_tokenization_spaces=False)
        for j in range(len(decoded_outputs)):
            end_idx = decoded_outputs[j].find('<|endoftext|>')
            if end_idx >= 0:
                decoded_outputs[j] = decoded_outputs[j][:end_idx]
        predictions.append(decoded_outputs)
        save_obj = {'idx': i, 'response': decoded_outputs}
        json_str = json.dumps(save_obj)
        file.write(json_str + '\n')
with open(f'{output_filename}.pkl', 'wb') as f:
    pickle.dump(predictions, f)