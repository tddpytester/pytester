import ast
import os
import sys
import torch
import logging
import pickle
import pandas as pd
from tqdm import tqdm
from handlers.code_processing import evaluation
from handlers.testing_util_v2 import functional_evaluation, split_test_cases, filtered_functional_evaluation, count_passing_testcase

#########################
######## config #########
test_dataset_path = 'datasets/APPS/apps_test_executable.csv' # ['datasets/APPS/apps_test_executable.csv','datasets/HumanEval/humaneval_test.csv','datasets/MBPP/mbpp_test.csv']
input_column = 'prompt_testcase' 
output_column = 'output_testcase'
result_mode = 'copilot' # ours, codex-codet, codex-api, baselines, copilot
pred_mode = 'single' # single, multi
n_pred_rank = 0 # 0 is the first beam 
n_test = 5 # if on single model
on_save = False

#### ours / copilot
output_dir = 'copilot_outputs' #'save/APPS/codet5-testcase-prompt_text-v2/checkpoint-10000', 'save/PyTester', 'copilot_outputs'
gen_testcase_path = output_dir + '/completion_4.pkl' #'/prediction-5_beam300.pkl' for most, '/prediction-5_beam.pkl' for codeT5-finetune, /copilot_completion_list.pkl / completion_0.pkl

# #### baselines (starcoder etc.) / codex-api 
# baseline_dir = 'baseline/save'
# baseline_name = '/openai_gpt35_apps_testcase_executable' # [openai_gpt35, starcoder, incoder-6B] '/openai_gpt35_apps_testcase_temp0_seed42_executable', 'starcoder_apps_testcase_beam_executable', 'openai_gpt35_apps_testcase_n5_seed42_0613_executable'
# output_dir = baseline_dir + baseline_name
# gen_testcase_path = baseline_dir + f'{baseline_name}.json'
# json_lines = False # default False / codex-api raw=True, processed=False

output_suffix = f'_rank{n_pred_rank}_{pred_mode}{n_test}' #f'_beam300_rank{n_pred_rank}_{pred_mode}{n_test}'

#########################
## Load Codex (CodeT gen data)
if result_mode == 'codex-codet':
    gen_testcase = pd.read_json(gen_testcase_path, lines=json_lines)
    predictions = list(gen_testcase['samples'])
elif result_mode == 'codex-api':
    gen_testcase = pd.read_json(gen_testcase_path, lines=json_lines)
    predictions = []
    for i in range(len(gen_testcase)):
        predictions.append([gen_testcase['response'][i]['choices'][n_pred_rank]['message']['content']])
elif result_mode == 'baselines':
    gen_testcase = pd.read_json(gen_testcase_path, lines=json_lines)
    predictions = list(gen_testcase['response'])
else:
    predictions = pickle.load(open(gen_testcase_path, 'rb'))

#########################
# Set logging
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = output_dir + f'/prediction_from_file_rank{n_pred_rank}_{pred_mode}{n_test}.log'
# log_file = output_dir + f'/prediction_from_file.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info(f'''evaluated_model_dir: {gen_testcase_path},
            test_data_dir: {test_dataset_path},
            input_column: {input_column},
            output_column: {output_column},
            output_dir: {output_dir},
            output_suffix: {output_suffix},
            on_save: {on_save},
            result_mode: {result_mode},
            pred_mode: {pred_mode},
            n_pred_rank: {n_pred_rank},
            n_test: {n_test}
            ''')
logger.info('load gen data finish.')

# Load the CSV test dataset 
test_df = pd.read_csv(test_dataset_path)
logger.info(f'# test data: {len(test_df)}')

test_inputs_text = list(test_df[input_column])
test_outputs_text = list(test_df[output_column])
fn_names=list(test_df['fn_name'])
prompt_testcases=list(test_df['prompt_testcase'])
prompt_codes=list(test_df['prompt_code'])
output_testcases=list(test_df['output_testcase'])
output_solutions=list(test_df['output_solution'])

######### Perfect Single ###########
if pred_mode == 'single':
    new_predictions = []
    for preds in predictions:
        p = []
        for pred in preds:
            if '\nassert' in pred:
                pred = "\nassert".join(pred.split("\nassert")[:n_test])
            p.append(pred)
        new_predictions.append(p)
    predictions = new_predictions
temp_n_pred_rank = 0 if result_mode == 'codex-api' else n_pred_rank
first_predictions = [pred[temp_n_pred_rank] for pred in predictions]

######## Evaluate the predictions against the expected outputs ##########
logger.info(f'*None-filtered assertions syntax results :{pred_mode}*')
em, es, mrr, parsable = evaluation(predictions, test_outputs_text, test_inputs_text, processed=False)
logger.info(f'Exact Match: {em}%')
logger.info(f'Edit Similarity: {es}%')
logger.info(f'MRR: {mrr}%')
logger.info('-----AST Parsable Rate-----')
for k in parsable:
    logger.info(f'{k}: {parsable[k]}%')

if result_mode != 'codex-codet':
    # filter assertions syntax
    filtered_predictions = []
    for i in range(len(predictions)):
        temp = []
        for p in predictions[i]:
            checked_assertions,_ = split_test_cases(p, fn_names[i], filter_syntax=True, add_test_call_solution=(result_mode=='ours'))
            temp.append("\n".join(checked_assertions)[7:].replace('test_call_solution(', 'call_solution('))
        filtered_predictions.append(temp)
    filtered_first_predictions = [pred[temp_n_pred_rank] for pred in filtered_predictions]
    if on_save:
        with open(output_dir + f'/filtered_prediction-5{output_suffix}.pkl', 'wb') as f:
            pickle.dump(filtered_predictions, f)
    logger.info(f'*Filtered assertions syntax results :{pred_mode}*')
    em, es, mrr, parsable = evaluation(filtered_predictions, test_outputs_text, test_inputs_text, processed=False)
    logger.info(f'Exact Match: {em}%')
    logger.info(f'Edit Similarity: {es}%')
    logger.info(f'MRR: {mrr}%')
    logger.info('-----AST Parsable Rate-----')
    for k in parsable:
        logger.info(f'{k}: {parsable[k]}%')

######### Perfect Multi ###########
logger.info('-----Functional Rate-----')
functional, coverage, mutation = functional_evaluation(prompts=prompt_codes,
                                   solutions=output_solutions,
                                   testcases=first_predictions,
                                   fn_names=fn_names,
                                   eval_coverage=True,
                                   eval_mutate=True,
                                   debug=False,
                                   on_guard=True,
                                   on_codet_result=(result_mode=='codex-codet'),
                                   add_test_call_solution=(result_mode=='ours'),
                                   filter_syntax=False)
if on_save:
    with open(output_dir + f'/evaluation_perfect{output_suffix}.pkl', 'wb') as f:
        pickle.dump([functional, coverage, mutation], f)
sys.stdin = sys.__stdin__
sys.stdout = sys.__stdout__
logger.info(f'*None-filtered assertions syntax results :{pred_mode}*')
for k in functional:
    logger.info(f'{k}: {functional[k]}%')
    print(f'{k}: {functional[k]}%')
logger.info(f'coverage: {sum(coverage) / len(coverage)}%')
print(f'coverage: {sum(coverage) / len(coverage)}%')
logger.info(f'Mutation score: {sum(mutation) / len(mutation)}%')
print(f'Mutation score: {sum(mutation) / len(mutation)}%')

######### Multi Filtered Syntax ###########
functional, coverage, mutation = functional_evaluation(prompts=prompt_codes,
                                   solutions=output_solutions,
                                   testcases=first_predictions,
                                   fn_names=fn_names,
                                   debug=False,
                                   on_guard=True,
                                   on_codet_result=(result_mode=='codex-codet'),
                                   add_test_call_solution=(result_mode=='ours'),
                                   filter_syntax=True,
                                   unique_name=output_dir[-3:]+output_suffix)
if on_save:
    with open(output_dir + f'/evaluation_syntax_filter{output_suffix}.pkl', 'wb') as f:
        pickle.dump([functional, coverage, mutation], f)
sys.stdin = sys.__stdin__
sys.stdout = sys.__stdout__
logger.info(f'*Filtered assertions syntax results :{pred_mode}*')
for k in functional:
    logger.info(f'{k}: {functional[k]}%')
    print(f'{k}: {functional[k]}%')
logger.info(f'coverage: {sum(coverage) / len(coverage)}%')
print(f'coverage: {sum(coverage) / len(coverage)}%')
logger.info(f'Mutation score: {sum(mutation) / len(mutation)}%')
print(f'Mutation score: {sum(mutation) / len(mutation)}%')

# ########### Multi Filtered Execution ##########
# if pred_mode == 'multi':
#     syntax, functional, coverage, mutation = filtered_functional_evaluation(prompts=prompt_codes,
#                                     solutions=output_solutions,
#                                     testcases=first_predictions,
#                                     fn_names=fn_names,
#                                     debug=False,
#                                     on_guard=True,
#                                     on_codet_result=(result_mode=='codex-codet'),
#                                     add_test_call_solution=(result_mode=='ours'),
#                                     filter_syntax=True,
#                                     unique_name=output_dir[-3:]+output_suffix)
#     if on_save:
#         with open(output_dir + f'/evaluation_exec_filter{output_suffix}.pkl', 'wb') as f:
#             pickle.dump([syntax, functional, coverage, mutation], f)
#     logger.info(f'*Filtered assertions error results: {pred_mode}*')
#     logger.info('Compilable Rate')
#     for k in syntax:
#         logger.info(f'{k}: {syntax[k]}%')
#         print(f'{k}: {syntax[k]}%')
#     logger.info('Functional Rate')
#     for k in functional:
#         logger.info(f'{k}: {functional[k]}%')
#         print(f'{k}: {functional[k]}%')
#     logger.info(f'Coverage: {sum(coverage) / len(coverage)}%')
#     print(f'Coverage: {sum(coverage) / len(coverage)}%')
#     logger.info(f'Mutation score: {sum(mutation) / len(mutation)}%')
#     print(f'Mutation score: {sum(mutation) / len(mutation)}%')

########### Count Passing Syntax / Execution Testcases ##########
if pred_mode == 'multi':
    count_syntax_passed_assertions, count_passed_assertions = count_passing_testcase(prompts=prompt_codes,
                                    solutions=output_solutions,
                                    testcases=first_predictions,
                                    fn_names=fn_names,
                                    debug=False,
                                    on_guard=True,
                                    on_codet_result=(result_mode=='codex-codet'),
                                    add_test_call_solution=(result_mode=='ours'),
                                    filter_syntax=True,
                                    unique_name=output_dir[-3:]+output_suffix)
    if on_save:
        with open(output_dir + f'/evaluation_passing_testcase{output_suffix}.pkl', 'wb') as f:
            pickle.dump([count_syntax_passed_assertions, count_passed_assertions], f)
    logger.info(f'*Passing testcase results: {pred_mode}*')
    logger.info(f'Syntax_passed_assertions: {sum(count_syntax_passed_assertions)*100 / len(count_syntax_passed_assertions)}%')
    logger.info(f'Passed_assertions: {sum(count_passed_assertions)*100 / len(count_passed_assertions)}%')

logger.info('finish program.')

