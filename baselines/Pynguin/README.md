# Pynguin

1. set PYNGUIN_DANGER_AWARE

### Shell

```
export PYNGUIN_DANGER_AWARE=x # set for the current shell and child processes
PYNGUIN_DANGER_AWARE=x <followed by executing code>
```

### In Python at top of your script:

```
import os
os.environ[PYNGUIN_DANGER_AWARE]="ok"
```

2. run

```
bash pynguin_humaneval.sh
```

3. copy data in "*_data" folder to "*_test" folder.

## Coverage

```
coverage run -m pytest humaneval_test_perfect/
coverage json -o coverage_humaneval_perfect.json
```

*Note:* 
Althought some test cases fail, the rest still pass for coverage, need to deduct manually (for perfect multi/single evaluation).
The as-is value is for filtered execution test cases evaluation.

## Mutation Score

```
cd humaneval_test_perfect
bash mutation_humaneval.sh
```