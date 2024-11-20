# PyTester: Deep Reinforcement Learning for Text-to-Testcase Generation :magic_wand:

![Screenshot 2023-09-28 at 13-09-04 TDD Without Tears Towards Test Case Generation from Requirements through Deep Reinforcement Learning - TDD_Without_Tears__Towards_Test_Case_Generation_from_Requirements_through_Deep_Reinforcement_Learning pdf](https://github.com/tddpytester/pytester/assets/146339482/0daf2abd-6846-4ee3-b772-bd9dae091768)


## :gear: Installation
```
pip install -r requirements.txt
```

## :running: Running
Please check the path of dataset and output directory before running.

### Finetuning
This script is for the *policy training step* in the paper. The model in this step is the CodeT5 baseline (see `baselines/Finetune CodeT5`).
```
python finetuning.py
```

### RL Training
This script is for the *policy optimizing step* in the paper. The model in this step is our PyTester (see results in the folder `results`).
```
python ppo_training.py
```

### Inference
This inference script is used for CodeT5 and PyTester models only. For other baseline models, please checkout the folder `baselines`.
```
python inference.py
```

### Evaluation
This evaluation script applies to all the models.
```
python evaluation.py
```

## :floppy_disk: Datasets and Models

- The processed APPS dataset (train and test sets) and the PyTester model can be accessed [here](https://drive.google.com/drive/folders/1ZPoCXkSitQmreo9CMj0fzsxYWT9k6zYz?usp=share_link). 

- The MBPP and HumanEval datasets are in the folder `datasets`. 

- The baseline result logs and predictions are in the folder `baselines`.

### Reference

If you use our code or PyTester, please cite our [PyTester Paper](https://arxiv.org/abs/2401.07576).

```
@article{takerngsaksiri2024tdd,
  title={TDD Without Tears: Towards Test Case Generation from Requirements through Deep Reinforcement Learning},
  author={Takerngsaksiri, Wannita and Charakorn, Rujikorn and Tantithamthavorn, Chakkrit and Li, Yuan-Fang},
  journal={arXiv preprint arXiv:2401.07576},
  year={2024}
}
```
