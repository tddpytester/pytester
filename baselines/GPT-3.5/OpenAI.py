import pandas as pd
import pickle
import json
import openai
import time
from tqdm import tqdm
import os.path
from tenacity import ( retry, stop_after_attempt, wait_random_exponential )  # for exponential backoff

filepath = '../../datasets/MBPP/mbpp_test.csv'
df_test = pd.read_csv(filepath)
input_column = 'prompt_testcase' #'prompt_testcase','prompt_testcase_llm'
output_filename = 'save/openai_gpt35_mbpp_testcase_n5_seed42_1106_full'
api_key = json.load(open('access_token.json'))
key = api_key['openai_access_token']

@retry(wait=wait_random_exponential(min=0.3, max=2), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

openai.api_key = key
openai_completions = []
curr_idx = 0
skip = False
if os.path.isfile(f'{output_filename}.json'):
    skip = True
    rfile = open(f"{output_filename}.json", "r")
    curr_idx = len(rfile.readlines())
    rfile.close()
file = open(f'{output_filename}.json', "a") 
for i in tqdm(range(len(df_test))):
    if i < curr_idx:
        continue
    if skip:
        print(f'continue at idx: {i}')
        skip = False
    completion = completion_with_backoff(
        # model="gpt-3.5-turbo-0613", #1106
        model="gpt-3.5-turbo",
        seed=42,
        n=5,
        max_tokens=300,
        messages=[
            {"role": "user", "content": df_test[input_column][i]}
        ]
    )
    openai_completions.append(completion)
    save_obj = {'idx': i, 'response': completion}
    json_str = json.dumps(save_obj)
    file.write(json_str + '\n')
    time.sleep(1)
file.close()
with open(f'{output_filename}.pkl', 'wb') as f:
    pickle.dump(openai_completions, f)