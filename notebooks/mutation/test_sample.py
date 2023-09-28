from sample import call_solution
import sys
import io
def adapt_call_solution(input_string):  # pragma: no cover
    sys.stdin = io.StringIO(input_string)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    call_solution()
    sys.stdin = sys.__stdin__
    sys.stdout = sys.__stdout__
    actual_output = captured_output.getvalue().strip()
    return actual_output

# check the correctness of the `call_solution` function
def test():
    assert adapt_call_solution('100') == '99'
    assert adapt_call_solution('48') == '48'
    assert adapt_call_solution('521') == '499'
    # assert adapt_call_solution('1') == '1'
    # assert adapt_call_solution('2') == '2'
    # assert adapt_call_solution('3') == '3'
    # assert adapt_call_solution('39188') == '38999'
    # assert adapt_call_solution('5') == '5'
    # assert adapt_call_solution('6') == '6'
    # assert adapt_call_solution('7') == '7'
    # assert adapt_call_solution('8') == '8'
    # assert adapt_call_solution('9') == '9'
    # assert adapt_call_solution('10') == '9'
    # assert adapt_call_solution('59999154') == '59998999'
    # assert adapt_call_solution('1000') == '999'
    # assert adapt_call_solution('10000') == '9999'
    # assert adapt_call_solution('100000') == '99999'
    # assert adapt_call_solution('1000000') == '999999'
    # assert adapt_call_solution('10000000') == '9999999'
    # assert adapt_call_solution('100000000') == '99999999'
    # assert adapt_call_solution('1000000000') == '999999999'
    # assert adapt_call_solution('10000000000') == '9999999999'
    # assert adapt_call_solution('100000000000') == '99999999999'
    # assert adapt_call_solution('1000000000000') == '999999999999'
    # assert adapt_call_solution('10000000000000') == '9999999999999'
    # assert adapt_call_solution('100000000000000') == '99999999999999'
    # assert adapt_call_solution('1000000000000000') == '999999999999999'
    # assert adapt_call_solution('10000000000000000') == '9999999999999999'
    # assert adapt_call_solution('100000000000000000') == '99999999999999999'
    # assert adapt_call_solution('1000000000000000000') == '999999999999999999'
    # assert adapt_call_solution('999999990') == '999999989'
    # assert adapt_call_solution('666666899789879') == '599999999999999'
    # assert adapt_call_solution('65499992294999000') == '59999999999999999'
    # assert adapt_call_solution('9879100000000099') == '8999999999999999'
    # assert adapt_call_solution('9991919190909919') == '9989999999999999'
    # assert adapt_call_solution('978916546899999999') == '899999999999999999'
    # assert adapt_call_solution('5684945999999999') == '4999999999999999'
    # assert adapt_call_solution('999999999999999999') == '999999999999999999'
    # assert adapt_call_solution('999999999999990999') == '999999999999989999'
    # assert adapt_call_solution('999999999999999990') == '999999999999999989'
    # assert adapt_call_solution('909999999999999999') == '899999999999999999'
    # assert adapt_call_solution('199999999999999999') == '199999999999999999'
    # assert adapt_call_solution('299999999999999999') == '299999999999999999'
    # assert adapt_call_solution('999999990009999999') == '999999989999999999'
    # assert adapt_call_solution('999000000001999999') == '998999999999999999'
    # assert adapt_call_solution('999999999991') == '999999999989'
    # assert adapt_call_solution('999999999992') == '999999999989'
    # assert adapt_call_solution('79320') == '78999'
    # assert adapt_call_solution('99004') == '98999'
    # assert adapt_call_solution('99088') == '98999'
    # assert adapt_call_solution('99737') == '98999'
    # assert adapt_call_solution('29652') == '28999'
    # assert adapt_call_solution('59195') == '58999'
    # assert adapt_call_solution('19930') == '19899'
    # assert adapt_call_solution('49533') == '48999'
    # assert adapt_call_solution('69291') == '68999'
    # assert adapt_call_solution('59452') == '58999'
    # assert adapt_call_solution('11') == '9'
    # assert adapt_call_solution('110') == '99'
    # assert adapt_call_solution('111') == '99'
    # assert adapt_call_solution('119') == '99'
    # assert adapt_call_solution('118') == '99'