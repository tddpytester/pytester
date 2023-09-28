# 0. imports
import os
import torch
import logging
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Config
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, set_seed
from fuzzywuzzy import fuzz
import ast
# import bitsandbytes as bnb
from handlers.code_processing import *
from handlers.testing_util_v2 import *
from handlers.python_terminal_command import PythonTerminalCommand

#########################
######## config #########
#pip install git+https://github.com/lvwerra/trl.git
model_name = 'codet5'
unique_name = f'{model_name}-testcase-v2' # f'{model_name}-single-clean-1'
model_dir = f'save/APPS/{unique_name}/checkpoint-10000' # f'save/APPS/{unique_name}/checkpoint-40060'
# unique_name = f'{model_name}-1' # f'{model_name}-single-clean-1'
# model_dir = f'save/APPS/{unique_name}/checkpoint-5500' # f'save/APPS/{unique_name}/checkpoint-40060'
tokenizer_dir = model_dir
train_dataset_path = 'dataset/APPS_new/apps_train_executable.csv'
input_column = 'prompt_testcase'
output_column = 'output_testcase'
output_dir = model_dir + '/PPO_functional-4'

single_unittest = True
gpu_num = '0'
# set_seed(42)

##########################
commander = PythonTerminalCommand(unique_name, random_unique_name=True)
base_model = {
    'transformers': AutoModelForCausalLMWithValueHead,
    'gpt2': AutoModelForCausalLMWithValueHead,
    'pycoder': AutoModelForCausalLMWithValueHead,
    'codegpt': AutoModelForCausalLMWithValueHead,
    'codet5': AutoModelForSeq2SeqLMWithValueHead
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
logger.info(f'''based_model: {model_name},
                model_dir: {model_dir},
                tokenizer_dir: {tokenizer_dir}, 
                train_data_dir: {train_dataset_path}, 
                input_column: {input_column},
                output_column: {output_column},
                output_dir: {output_dir}'''
            )
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load a pretrained model
if model_name == 'transformers':
    config = GPT2Config()
    tokenizer = AutoTokenizer.from_pretrained('gpt2', truncation_side='left', do_lower_case=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = base_model[model_name](config).to(device)
    model_ref = base_model[model_name](config).to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, truncation_side='left', do_lower_case=False)
    if model_name == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    model = base_model[model_name].from_pretrained(model_dir, v_head_init_strategy='orthogonal', v_head_initializer_range=1.0).to(device)
    model_ref = base_model[model_name].from_pretrained(model_dir).to(device)
logger.info('loaded model and tokenizer sucessfully.')

# 2. Load and encode dataset
# Load the CSV 
train_df = pd.read_csv(train_dataset_path)
logger.info(f'# train data: {len(train_df)}')

# Convert the tokenized inputs and outputs into a PyTorch dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, prompt_text, fn_names, prompt_testcase, prompt_code, output_testcase, output_solution):
        self.inputs = inputs
        self.fn_names = fn_names
        self.outputs = outputs
        self.prompt_text = prompt_text
        self.prompt_testcase = prompt_testcase
        self.prompt_code = prompt_code
        self.output_testcase = output_testcase
        self.output_solution = output_solution

    def __getitem__(self, idx):
        return {
            "query": self.prompt_text[idx],
            "fn_name": self.fn_names[idx],
            "prompt_testcase": self.prompt_testcase[idx],
            "prompt_code": self.prompt_code[idx],
            "output_testcase": self.output_testcase[idx],
            "output_solution": self.output_solution[idx],
            "input_ids": torch.tensor(self.inputs["input_ids"][idx]),
            "attention_mask": torch.tensor(self.inputs["attention_mask"][idx]),
            "decoder_input_ids": torch.tensor(self.outputs["input_ids"][idx]),
            "decoder_attention_mask": torch.tensor(self.outputs["attention_mask"][idx]),
            "labels": torch.tensor(self.outputs["input_ids"][idx]),
        }

    def __len__(self):
        return len(self.inputs["input_ids"])
    
# Tokenize the inputs and outputs 
# Convert the tokenized inputs and outputs into a PyTorch dataset
train_inputs = tokenizer(list(train_df[input_column]), padding=True, truncation=True, max_length=512)
train_outputs = tokenizer(list(train_df[output_column]), padding=True, truncation=True, max_length=512)
train_dataset = MyDataset(inputs=train_inputs,
                          outputs=train_outputs,
                          prompt_text=list(train_df[input_column]),
                          fn_names=list(train_df['fn_name']),
                          prompt_testcase=list(train_df['prompt_testcase']),
                          prompt_code=list(train_df['prompt_code']),
                          output_testcase=list(train_df['output_testcase']),
                          output_solution=list(train_df['output_solution']))
logger.info('loaded dataset sucessfully.')

# 3. initialize trainer
# default_params = {
#         "lr": 1e-5,
#         "adap_kl_ctrl": True, 
#         "init_kl_coef": 0.2, 
#         "target": 6,
#         "horizon":10000,
#         "gamma":1,
#         "lam":0.95,
#         "cliprange": 0.2,
#         "cliprange_value": 0.2,
#         "vf_coef": 0.1, 
#         "batch_size": 256,
#         "forward_batch_size": None,
#         "ppo_epochs": 4
# }

# the critic share the same parameters (backbone) with the actor
ppo_config = {
    'remove_unused_columns': False,
    'batch_size': 8,
    # 'forward_batch_size': 64,
    'mini_batch_size': 4,
    'gradient_accumulation_steps': 8,
    'max_grad_norm': 0.5,
    'early_stopping': True, # default False
    'ppo_epochs': 1, # default 4
    'learning_rate': 1e-5, # default 1e-5
    'adap_kl_ctrl': False, # default True
    'init_kl_coef': 0.2, # 0.2, # default 0.2
    'vf_coef': 0.1, # default 0.1
    # 'target': 6, # default 6
    'cliprange': 0.1, # default 0.2 for ppo clip
    'cliprange_value': 0.05, # default 0.2 for loss clip
    'log_with':"wandb",
    'optimize_cuda_cache': True
}
import wandb
wandb.init('PPO_training')
config = PPOConfig(**ppo_config)
logger.info(f'ppo_config: {config}')
# optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.learning_rate)
ppo_trainer = PPOTrainer(
    config, 
    model, 
    model_ref, 
    tokenizer, 
    dataset=train_dataset, 
#     optimizer=optimizer
)
generation_kwargs = {
    "min_new_tokens": 5,
    "temperature": 0.7,
    # "top_k": 0.0,
    "top_p": 0.7,
    "do_sample": True,
    "max_new_tokens": 100,
    "pad_token_id": tokenizer.eos_token_id,
    # "eos_token_id": tokenizer('\n').input_ids[1:],
}

logger.info('training start.')
logger.info(f'total batch: {len(ppo_trainer.dataloader)}')

step = 0
for epoch in range(3):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = [r for r in batch['input_ids']]
        # 4. generate model response
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # 5. define a reward for response
        # (this could be any reward such as human feedback or output from another model)
        # reward = [torch.tensor(1.0)]
        # rewards = [torch.tensor(1.0) for output in batch['response']] # return 1
        # rewards = [torch.tensor(float(fuzz.ratio(output, gt))/100) for output,gt in zip(batch['response'],batch['query'])] # edit similarity
        # rewards = [torch.tensor(1.0) if output.strip() == gt.strip() else torch.tensor(0.0) for output,gt in zip(batch['response'],batch['query'])] # exact match
        rewards = []
        for input_text, pred_text, fn_name, prompt_code, prompt_testcase,output_solution,output_testcase in zip(batch['query'],batch['response'],batch['fn_name'],batch['prompt_code'],batch['prompt_testcase'],batch['output_solution'],batch['output_testcase']):
            ### Parsable ###
            # full_code = input_text + pred_text
            # _,_,compilable = commander.process_code_string(code_processing.clean_to_code(full_code),cmd='compiler')

            # _, compilable = test_parsable_ast(full_code)
            reward = 0.0
            # reward = 1.0 if compilable else -1.0
            reward += 1.0 if fn_name in pred_text else -1.0
            reward += 0.5 if '==' in pred_text else 0.0
            reward += -0.5 if len(pred_text) < 5 else 0.0
            # rewards.append(torch.tensor(reward)) 

            # '''
            # exec assert correctly = 5.0
            # exec AssertError = 0.0
            # exec OtherError = -1.0
            # compile Error = -2.0
            # '''
            # tmp_rewards,_ = test_function([pred_text], [output_solution], single_unittest=single_unittest) # ver1 functional
            rewards_function={
                      'pb_error':-3,
                      'syntax_error':-2,
                      'runtime_error':-1,
                      'assert_error':-0.3,
                      'executable':5
                      }
            tmp_rewards, errors = test_function(prompts=[prompt_code],
                                         solutions=[output_solution], 
                                         testcases=[pred_text], 
                                         fn_names=[fn_name],
                                         debug=False,
                                         rewards=rewards_function)
            reward += tmp_rewards[0]

            # code coverage reward
            if not errors[0]:
                script = transform_to_input(prompt=prompt_code,
                                         solution=output_solution, 
                                         testcase=pred_text, 
                                         fn_name=fn_name,
                                         filter_syntax=True)
                err, coverage_output, executable = commander.process_coverage_test(script)
                coverage_score = int(coverage_output.split()[-1][:-1])
                if executable:
                    reward += coverage_score

            rewards.append(torch.tensor(reward))

        # 6. train model with ppo
        train_stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(train_stats,batch,rewards)

        # save checkpoint locally
        curr_mean_reward = torch.tensor(rewards).mean()
        if step == 0: 
            highest_reward = curr_mean_reward
        step += 1
        if curr_mean_reward == highest_reward:
            save_point_dir = output_dir + f'/step-same-{step}'
            if not os.path.exists(save_point_dir):
                os.makedirs(save_point_dir)
            model.save_pretrained(save_point_dir)
            tokenizer.save_pretrained(save_point_dir)
        elif curr_mean_reward > highest_reward:
            # if the best reward ever seen, save
            highest_reward = curr_mean_reward
            save_point_dir = output_dir + f'/step-better-{step}'
            if not os.path.exists(save_point_dir):
                os.makedirs(save_point_dir)
            model.save_pretrained(save_point_dir)
            tokenizer.save_pretrained(save_point_dir)
        elif step % 10 == 0:
            save_point_dir = output_dir + f'/step-{step}'
            if not os.path.exists(save_point_dir):
                os.makedirs(save_point_dir)
            model.save_pretrained(save_point_dir)
            tokenizer.save_pretrained(save_point_dir)

    save_point_dir = output_dir + f'/step-ep-{epoch}'
    if not os.path.exists(save_point_dir):
        os.makedirs(save_point_dir)
    model.save_pretrained(save_point_dir)
    tokenizer.save_pretrained(save_point_dir)

# ####### use  trained model ########

# # load the model from the Hub
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("my-fine-tuned-model-ppo")

# # continue the training
# from trl.model import AutoModelForCausalLMWithValueHead
# model = AutoModelForCausalLMWithValueHead.from_pretrained("my-fine-tuned-model-ppo")
