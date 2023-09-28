import faulthandler
import signal
import ast
import sys
sys.set_int_max_str_digits(99999999)
from io import StringIO 

# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

# stuff for setting up signal timer
class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    print("alarm went off")
    #return
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)
timeout = 10  # seconds

def check_test_case_syntax(test_case):
    if len(test_case.strip()) < 1:
        return False
    if 'assert' not in test_case:
        return False
    try:
        ast.parse(test_case)
        return True
    except Exception:
        return False

def split_test_cases(output_testcase, fn_name, limit=5):
    if fn_name.strip() == 'call_solution':
        split_asserts = [f'assert test_{part}'.strip() for part in f'assert {output_testcase}'.split('assert ') if (fn_name.strip() in part) and len(part.strip()) > 0]
    else:
        split_asserts = [f'assert {part}'.strip() for part in f'assert {output_testcase}'.split('assert ') if (fn_name.strip() in part) and len(part.strip()) > 0]
    checked_assertions = [i for i in split_asserts if check_test_case_syntax(i)][:limit]
    return checked_assertions

def test_function(preds, solutions, single_unittest, debug=False):
    results = []
    errors = []
    for pred, solution in zip(preds,solutions):
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple, Optional\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\nfrom datetime import datetime\nimport bisect\nfrom functools import cmp_to_key\nfrom numpy import inf\n"
        sol += solution
        signal.alarm(timeout)

        ## check gt solution syntax ##
        try:
            ast.parse(sol)
            # signal.alarm(0) # reset the alarm
        except Exception as e:
            signal.alarm(0)
            if debug:
                print(sol)
                print(f"type 0 gt_solution compilation error = {e}")
            results.append(-3)
            errors.append(err_name)
            continue
        ## check solution + unit test syntax##
        try:
            end_idx = pred.find('\nassert') if (single_unittest and pred.find('\nassert') != -1)  else len(pred)
            if "class Solution" in sol:
                sol_with_test = sol + '\nsol = Solution()\nassert sol.' + pred[:end_idx]
            else:
                sol_with_test = sol + '\nassert ' + pred[:end_idx]
            ast.parse(sol_with_test)
        except Exception as e:
            # case 1:
            # predicted testcase not parsable 
            signal.alarm(0)
            err_name = type(e).__name__
            if debug:
                print(sol_with_test)
                print(f"type 1 compilation error = {err_name}")
            results.append(-2)
            errors.append(err_name)
            continue

        ## check runtime ##
        with Capturing() as output:
            try:
                # case2:
                # predicted testcase executed til the end correctly by gt
                compiled_code = compile(sol_with_test, '<string>', 'exec')
                namespace = {}
                exec(compiled_code, namespace)
                results.append(1)
                errors.append(None)
            except Exception as e:
                signal.alarm(0)
                faulthandler.disable()
                err_name = type(e).__name__
                print(f'Err name: {err_name}')
                print(e)
                print()
                if debug:
                    print(f"type 2 runtime error or time limit exceeded error = {err_name}")
                if 'AssertionError' in err_name:
                    # case 3:
                    # predicted testcase has wrong input-output pair
                    results.append(0)
                else:
                    # case 4:
                    # any other errors (e.g. runtime error, indentation)
                    results.append(-1)
                errors.append(err_name)
                continue
        faulthandler.disable()
        signal.alarm(0)
    return results, errors
