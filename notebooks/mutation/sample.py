def call_solution():
    r"""
    Anton has the integer x. He is interested what positive integer, which doesn't exceed x, has the maximum sum of digits.
    
    Your task is to help Anton and to find the integer that interests him. If there are several such integers, determine the biggest of them. 
    
    
    -----Input-----
    
    The first line contains the positive integer x (1 ≤ x ≤ 10^18) — the integer which Anton has. 
    
    
    -----Output-----
    
    Print the positive integer which doesn't exceed x and has the maximum sum of digits. If there are several such integers, print the biggest of them. Printed integer must not contain leading zeros.
    
    
    -----Examples-----
    Input
    100
    
    Output
    99
    
    Input
    48
    
    Output
    48
    
    Input
    521
    
    Output
    499
    """
    num = list(map(int, input()))
    best = num[:]
    for i in range(-1, -len(num) - 1, -1):
        if num[i] == 0:
            continue
        num[i] -= 1
        for j in range(i + 1, 0):
            num[j] = 9
        if sum(num) > sum(best):
            best = num[:]
    s = ''.join(map(str, best)).lstrip('0')
    print(s)
