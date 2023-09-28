import pandas as pd
import numpy as np
import os
import logging
import torch
# from unixcoder import UniXcoder
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, \
                        GPT2Config, GPT2Model, RobertaModel, PLBartForCausalLM, \
                        Trainer, TrainingArguments, \
                        DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
import evaluate

#########################
######## config #########
# export CUDA_VISIBLE_DEVICES=0
model_name = 'codet5-large'
is_shuffle = True

train_dataset_path = 'dataset/APPS_new/apps_train_4450.csv'
# eval_dataset_path = 'dataset/APPS/apps_processed_eval.csv' # not use
input_column = 'prompt_testcase_llm'
output_column = 'output_testcase'
output_dir = 'save/APPS/codet5-testcase-prompt_llm-v2'

gpu_num = '0'

#########################
model_name = model_name.lower()
base_model = {
    'pycoder': AutoModelForCausalLM,
    'codegpt': AutoModelForCausalLM,
    'transformers': GPT2Model,
    'gpt2': AutoModelForCausalLM,
    'codet5-small': AutoModelForSeq2SeqLM,
    'codet5-base': AutoModelForSeq2SeqLM,
    'codet5-large': AutoModelForSeq2SeqLM,
    # 'unixcoder': UniXcoder,
    'plbart': PLBartForCausalLM,
}
base_checkpoint = {
    'pycoder': 'Wannita/PyCoder',
    'codegpt': 'microsoft/CodeGPT-small-py',
    'transformers': 'gpt2_no_pretrain_weight',
    'gpt2': 'gpt2',
    'codet5-small': 'Salesforce/codet5-small',
    'codet5-base': 'Salesforce/codet5-base',
    'codet5-large': 'Salesforce/codet5-large',
    'unixcoder': 'microsoft/unixcoder-base',
    'plbart': 'uclanlp/plbart-base',
}
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set logging
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = output_dir + '/train.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
logger.info(f'''based_model_dir: {base_checkpoint[model_name]},
                tokenizer_dir: {base_checkpoint[model_name]}, 
                train_data_dir: {train_dataset_path}, 
                eval_data_dir: {train_dataset_path}, 
                input_column: {input_column},
                output_column: {output_column},
                output_dir: {output_dir},
                is_shuffle: {is_shuffle}'''
            )
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the tokenizer, model
if model_name == 'transformers':
    config = GPT2Config()
    tokenizer = AutoTokenizer.from_pretrained('gpt2', truncation_side='left', do_lower_case=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = base_model[model_name](config)
elif model_name == 'unixcoder':
    config =AutoConfig.from_pretrained(base_checkpoint[model_name])
    config.is_decoder = True
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint[model_name], truncation_side='left', do_lower_case=False)
    encoder = RobertaModel.from_pretrained(base_checkpoint[model_name],config=config) 
    model=Seq2Seq(encoder=encoder,decoder=encoder,config=config,
                  beam_size=5,max_length=512,
                  sos_id=tokenizer.cls_token_id,eos_id=[tokenizer.sep_token_id])
else:
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint[model_name], truncation_side='left', do_lower_case=False) # padding_side='left'
    if model_name == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    if model_name == 'plbart':
        model = base_model[model_name].from_pretrained(base_checkpoint[model_name], add_cross_attention=False)
        assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
    else:
        model = base_model[model_name].from_pretrained(base_checkpoint[model_name])
logger.info('loaded model and tokenizer sucessfully.')

# Load the CSV 
train_df = pd.read_csv(train_dataset_path)
# eval_df = pd.read_csv(eval_dataset_path) 
# print("null input:", len(train_df[input_column].dropna()))
# print("null output:", len(train_df[output_column].dropna()))
train_df, eval_df = train_test_split(train_df, test_size=0.1, random_state=42, shuffle=is_shuffle)

# Convert the tokenized inputs and outputs into a PyTorch dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs["input_ids"][idx]),
            "attention_mask": torch.tensor(self.inputs["attention_mask"][idx]),
            # "decoder_input_ids": torch.tensor(self.outputs["input_ids"][idx]),
            # "decoder_attention_mask": torch.tensor(self.outputs["attention_mask"][idx]),
            "labels": torch.tensor(self.outputs["input_ids"][idx]),
        }

    def __len__(self):
        return len(self.inputs["input_ids"])

# Tokenize the inputs and outputs 
# Convert the tokenized inputs and outputs into a PyTorch dataset
train_inputs = tokenizer(list(train_df[input_column]), padding=True, truncation=True, max_length=512)
# train_df[output_column] = train_df[output_column].apply(lambda x: x.replace('call_solution(','test_call_solution('))
train_outputs = tokenizer(list(train_df[output_column]), padding=True, truncation=True, max_length=512)
train_dataset = MyDataset(train_inputs, train_outputs)

eval_inputs = tokenizer(list(eval_df[input_column]), padding=True, truncation=True, max_length=512)
# eval_df[output_column] = eval_df[output_column].apply(lambda x: x.replace('call_solution(','test_call_solution('))
eval_outputs = tokenizer(list(eval_df[output_column]), padding=True, truncation=True, max_length=512)
eval_dataset = MyDataset(eval_inputs, eval_outputs)
logger.info('loaded dataset sucessfully.')

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5, #1.5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    # predict_with_generate=True,
    num_train_epochs=20,
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="epoch",
    eval_accumulation_steps=1,
    evaluation_strategy="epoch",
    warmup_steps=1000,
    fp16=True,
    save_total_limit=10,
    optim="adamw_torch",
    lr_scheduler_type = "inverse_sqrt"
)
logger.info(f'''training_args: lr={training_args.learning_rate}, 
            batch_size={training_args.per_device_train_batch_size}, 
            epoch={training_args.num_train_epochs}, 
            gradient_accumulation_steps={training_args.gradient_accumulation_steps}, 
            warmup_steps={training_args.warmup_steps}, 
            weight_decay={training_args.weight_decay},
            optim={training_args.optim},
            lr_scheduler_type={training_args.lr_scheduler_type},
            fp16={training_args.fp16}
            ''')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer) if 'codet5' in model_name else DataCollatorForLanguageModeling(tokenizer, mlm=False),
    tokenizer=tokenizer,
)

trainer.train()