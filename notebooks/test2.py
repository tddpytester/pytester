import sys
import io

def call_solution():    
    from collections import defaultdict as dd
    import math
    def nn():
        return int(input())

    def li():
        return list(input())

    def mi():
        return list(map(int, input().split()))

    def lm():
        return list(map(int, input().split()))


    n, q=mi()

    ints=[]


    for _ in range(q):
        st, end=mi()
        ints.append((st,end))


    coverage=[10]+[0]*n

    for st, end in ints:
        for i in range(st,end+1):
            coverage[i]+=1

    total=-1

    for val in coverage:
        if not val==0:
            total+=1

    singlecount=0
    doublecount=0

    singles=[0]*(n+1)
    #print(total)
    doubles=[0]*(n+1)
    for i in range(len(coverage)):
        #print(i,singles)
        if coverage[i]==1:
            singlecount+=1
        if coverage[i]==2:
            doublecount+=1
        singles[i]=singlecount
        doubles[i]=doublecount
    maxtotal=0
    for i in range(len(ints)):
        for j in range(i+1, len(ints)):
            st1=min(ints[i][0],ints[j][0])
            end1=min(ints[i][1],ints[j][1])
            st2, end2=max(ints[i][0],ints[j][0]), max(ints[i][1],ints[j][1])
            #assume st1<=st2
            if end1<st2:
                curtotal=total-(singles[end1]-singles[st1-1])-(singles[end2]-singles[st2-1])
            elif end1<end2:
                curtotal=total-(singles[st2-1]-singles[st1-1])-(doubles[end1]-doubles[st2-1])-(singles[end2]-singles[end1])
            else:
                curtotal=total-(singles[st2-1]-singles[st1-1])-(doubles[end2]-doubles[st2-1])-(singles[end1]-singles[end2])
            maxtotal=max(maxtotal,curtotal)

    print(maxtotal)

def test_call_solution(input_string, expected_result):
    # Redirect sys.stdin to use the input string
    sys.stdin = io.StringIO(input_string)
    # Redirect sys.stdout to capture the printed output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function that takes input from standard input
    call_solution()

    # Restore the original sys.stdin and sys.stdout
    sys.stdin = sys.__stdin__
    sys.stdout = sys.__stdout__

    # Get the captured printed output
    actual_output = captured_output.getvalue().strip()

    # Perform the assertion
    assert actual_output == expected_result, f"Expected: {expected_result}, Got: {actual_output}"

def test_call_solution(input_string):
    sys.stdin = io.StringIO(input_string)
    captured_output = io.StringIO()
    sys.stdout = captured_output
    call_solution()
    sys.stdin = sys.__stdin__
    sys.stdout = sys.__stdout__
    actual_output = captured_output.getvalue().strip()
    return actual_output

# Test case 1
input_string1 = "7 5\n1 4\n4 5\n5 6\n6 7\n3 5\n"
# input_string1 = "4 3\n1 1\n2 2\n3 4\n"
expected_result1 = '7\n'
# test_call_solution(input_string1, expected_result1)

assert test_call_solution(input_string1) == expected_result1