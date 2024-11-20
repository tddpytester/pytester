import subprocess
import os
import re
import json
from random import randint

class PythonTerminalCommand:
    def __init__(self, file_unique_name, random_unique_name=True) -> None:
        self.temp_path = '.'
        self.file_unique_name = file_unique_name
        self.text2cmd = {
            'coverage_run': 'coverage run ',
            'coverage_report': 'coverage report ',
            'coverage_json': 'coverage json ',
            'compiler': 'python3 -m py_compile ',
            'linter_err': 'pylint -E ',
            'linter': 'pylint ',
            'mutate_score': f'mut.py --runner pytest '
        }
        if random_unique_name:
            self.file_unique_name += str(randint(1000,999999))
    def run_command(self, filepath, cmd):
        cmd = self.text2cmd[cmd] + filepath
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, shell=True)
        error = [i.decode('utf-8') for i in proc.stderr.readlines()]
        err = '\n'.join(error)
        output = [i.decode('utf-8') for i in proc.stdout.readlines()]
        op = '\n'.join(output)
        return err, op

    def process_code_string(self, code_string, cmd, print_error=False, delete_file=True):
        tempfile_path = self.temp_path + f'/test{self.file_unique_name}' + '.py'
        try:
            with open(tempfile_path, "w+", encoding = 'utf-8') as tf:
                tf.write(code_string)
            error, output = self.run_command(tempfile_path, cmd=cmd)
        finally:
            if delete_file:
                os.remove(tempfile_path)
            
        if print_error:
            print("Error: ", error)

        if error:
            return error, output, False
        else:
            return error, output, True

    def process_code_file(self, file_path, cmd, print_error = False):
        error, output = self.run_command(file_path, cmd=cmd)
        if print_error:
            print("Error: ", error)
        if error:
            return error, output, False
        else:
            return error, output, True
    
    def process_coverage_test(self, script_string, print_error=False, delete_file=True):
        temptest_path = self.temp_path + f'/test_coverage_{self.file_unique_name}.py'
        tempcov_path = self.temp_path + f'/.coverage_{self.file_unique_name}'
        tempjson_path = self.temp_path + f'/coverage_{self.file_unique_name}.json'
        coverage_score = 0
        try:
            with open(temptest_path, "w+", encoding = 'utf-8') as tf:
                tf.write(script_string)
            cov_run_arg = f"--data-file='{tempcov_path}' {temptest_path}"
            error, _ = self.run_command(cov_run_arg, cmd='coverage_run')
            if not error:
                cov_json_arg = f"--data-file='{tempcov_path}' -o {tempjson_path}"
                error, _ = self.run_command(cov_json_arg, cmd='coverage_json')
                with open(tempjson_path, "r", encoding = 'utf-8') as f:
                    cov_data = json.load(f)
                try:
                    coverage_score = int(cov_data['totals']['percent_covered'])
                except Exception as e:
                    coverage_score = 0
                    error = True
                    err_name = type(e).__name__
                    print(f'coverage score error: {err_name} -- detail: {e}')
        finally:
            if delete_file:
                os.remove(temptest_path)
                os.remove(tempcov_path)
                os.remove(tempjson_path)
        if print_error:
            print("Error: ", error)

        if error:
            return error, coverage_score, False
        else:
            return error, coverage_score, True
    
    def process_mutation_test_score(self, function_string, test_function_string, delete_file=True):
        tempfile_function_path = self.temp_path + f'/{self.file_unique_name}' + '.py'
        tempfile_function_test_path = self.temp_path + f'/test_{self.file_unique_name}' + '.py'
        try:
            with open(tempfile_function_path, "w+", encoding = 'utf-8') as tf:
                tf.write(function_string)
            with open(tempfile_function_test_path, "w+", encoding = 'utf-8') as tf:
                tf.write(test_function_string)
            tempfile_path = f'--target {self.file_unique_name} --unit-test test_{self.file_unique_name}'
            error, output = self.run_command(tempfile_path, cmd='mutate_score')
            if not error:
                if 'Tests failed' in output or 'Mutation score' not in output:
                    mutation_score = 0.0
                else:
                    mutation_score = float(re.findall(r'[\n\r]*Mutation score.*: [ \t]*([^\n\r]*)', output)[0][:-1])
            else:
                # print(error)
                mutation_score = 0.0
        except Exception as e:
            err_name = type(e).__name__
            print(f'mutate score error: {err_name} -- detail: {e}')
            error = True
            mutation_score = 0.0
        finally:
            if delete_file:
                os.remove(tempfile_function_path)
                os.remove(tempfile_function_test_path)
        return error, mutation_score
        