import pandas as pd
import logging
from tqdm import tqdm
from exec_program import functional_evaluation, test_function
from _execution_from_codeT import check_correctness_with_test_cases
logging.basicConfig(filename='exec.log', level=logging.INFO)

# df_apps = pd.read_csv('apps_small_sample.csv')
print('start')
# df_apps = pd.read_csv('apps_test_3765.csv')
# df_apps = pd.read_csv('/home/wannita/HDD18TB/active_repo/pyRL/text-to-testcase/dataset/APPS_new/apps_train_4450.csv')
df_apps = pd.read_csv('/home/wannita/HDD18TB/active_repo/pyRL/text-to-testcase/dataset/APPS_new/apps_test_executable.csv') # apps_test_3765.csv')
# results = functional_evaluation(df_apps['prompt_code'][500:800], df_apps['output_solution'][500:800], df_apps['output_testcase'][500:800], df_apps['fn_name'][500:800], debug=True)
# results = functional_evaluation(df_apps['prompt_code'], df_apps['output_solution'], df_apps['output_testcase'], df_apps['fn_name'], debug=True)
# logging.info(results)
# print(results)

results = {'parsable_rate' : 0,
'executable_all' : 0,
'executable_some' : 0,
'executable' : 0,
'tc_count' : 0}
for idx in tqdm(range(len(df_apps))):
    task_id,test_cases,completion,passed,result  = check_correctness_with_test_cases(idx, df_apps['prompt_code'][idx], df_apps['output_solution'][idx], df_apps['output_testcase'][idx], df_apps['fn_name'][idx], 5)
    results['parsable_rate'] += 1 if passed else results['parsable_rate']
    results['executable_all'] += 1 if all(result) else results['executable_all']
    results['executable_some'] += 1 if any(result) else results['executable_some']
    # results['executable'] += sum(result)
    # results['tc_count'] += len(result)
logging.info(results)

# # for filter executable-only data
# _, errors = test_function(df_apps['prompt_code'][idx], df_apps['output_solution'], df_apps['output_testcase'], df_apps['fn_name'], debug=False)
# df_apps['exec_error'] = errors
# df_apps[['problem_id','difficulty','fn_name','solutions','asserts','solution1','prompt_code', 'prompt_testcase', 'output_testcase','output_solution', 'exec_error']].to_csv('apps_test_3765.csv', index=False)
# df_apps[df_apps['exec_error'].isnull()][['problem_id','difficulty','fn_name','solutions','asserts','solution1','prompt_code', 'prompt_testcase', 'output_testcase','output_solution', 'exec_error']].to_csv('apps_test_executable.csv',index=False)