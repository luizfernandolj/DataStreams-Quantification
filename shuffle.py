import pandas as pd
import os

a = "AedesQuinx"

path = f"{os.getcwd()}/datasets/test/{a}.test.csv"

df = pd.read_csv(f"datasets/test/{a}.test.data")
df = df.groupby(["context"])
d1 = df.get_group(1).sample(frac=1).reset_index(drop=True)
d2 = df.get_group(2).sample(frac=1).reset_index(drop=True)
df_final = pd.concat([d1, d2], ignore_index=True)

df_final.to_csv(path, index=False)
