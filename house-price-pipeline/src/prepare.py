import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open("params.yaml"))["data"]
df = pd.read_csv(p["raw_path"]).drop(columns=p["drop_cols"], errors="ignore")

train, test = train_test_split(df, test_size=p["test_size"], random_state=p["seed"])

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print(f"train={len(train)} test={len(test)}")