from handlers.testing_util_v2 import test_mutation
import pandas as pd

i = 10
test_dataset_path = 'dataset/APPS_new/apps_test_executable.csv'
test_df = pd.read_csv(test_dataset_path)
fn_names=list(test_df['fn_name'])[:i]
prompt_testcases=list(test_df['prompt_testcase'])[:i]
prompt_codes=list(test_df['prompt_code'])[:i]
output_testcases=list(test_df['output_testcase'])[:i]
output_solutions=list(test_df['output_solution'])[:i]
score = test_mutation(prompt_codes, output_solutions, output_testcases, fn_names, on_codet_result=False, filter_syntax=True, add_test_call_solution=True, unique_name='')
print('mutation score: ', score)