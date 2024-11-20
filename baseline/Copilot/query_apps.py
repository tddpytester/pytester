import json
import requests


def post(body):
    res = requests.post("http://localhost:8080/api", data=json.dumps(body))
    return res.content.decode("utf-8")  # convert bytes to string


if __name__ == "__main__":
    import os
    import pickle
    import time
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv("../dataset/APPS/apps_test_executable.csv")
    nrows = len(df)
    print(f"# rows: {len(df)}")
    for repeat in range(0,5):
        completions = []
        for i, row in tqdm(df.iterrows(), total=nrows):
            prompt = row["prompt_testcase_llm"]
            body = {"prompt": prompt}
            completion = post(body)
            completions.append([completion])
            time.sleep(4)
        os.makedirs("copilot_outputs/apps", exist_ok=True)
        with open(f"copilot_outputs/apps/completion_{repeat}.pkl", "wb") as f:
            pickle.dump(completions, f)
        with open(f"copilot_outputs/apps/completion_{repeat}.txt", "w") as f:
            for completion in completions:
                f.write(completion + "\n")

