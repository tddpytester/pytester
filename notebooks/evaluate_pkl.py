import os
import sys
import pandas as pd
import pickle
# from exec_program import functional_evaluation
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append('../handlers/')
from testing_util_v2 import functional_evaluation
import logging
log_file = 'prediction.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger.addHandler(logging.StreamHandler())

# pred_tc = pickle.load(open('/home/wannita/HDD18TB/active_repo/pyRL/text-to-testcase/save/APPS/codet5-testcase-v2/checkpoint-10000/prediction-5.pkl', 'rb'))
# pred_tc = pickle.load(open('/home/wannita/HDD18TB/active_repo/pyRL/text-to-testcase/save/APPS/codet5-testcase-v2/checkpoint-10000/PPO_functional-1/step-65/prediction-5.pkl', 'rb'))
# pred_tc = pickle.load(open('/home/wannita/HDD18TB/active_repo/pyRL/text-to-testcase/save/APPS/codet5-testcase-v2/checkpoint-10000/PPO_functional-2/step-887/prediction-5.pkl', 'rb'))

file_name = '../dataset/APPS_new/apps_train_executable.csv'
df_test = pd.read_csv(file_name)
print('finish loading data.')
logger.info(f'execute data: {file_name}')

# pred_tc_1 = [t[0] for t in pred_tc]
# results = functional_evaluation(prompts=df_test['prompt_code'], 
#                       solutions=df_test['output_solution'],
#                       testcases=pred_tc_1,
#                       fn_names=df_test['fn_name'],
#                       debug=False)
functional, coverage = functional_evaluation(prompts=df_test['prompt_code'][:],
            solutions=df_test['output_solution'][:],
            testcases=df_test['output_testcase'][:],
            fn_names=df_test['fn_name'][:],
            debug=False)
logger.info(functional)
print(functional)
logger.info(f'coverage: {sum(coverage) / len(coverage)}%')
print(f'coverage: {sum(coverage) / len(coverage)}%')