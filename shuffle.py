import pandas as pd
import os

a = "wine"

path = f"{os.getcwd()}/datasets/training/{a}.train"

df = pd.read_csv(f"{path}.data")
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv(f"{path}.csv", index=False)
