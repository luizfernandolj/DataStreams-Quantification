import pandas as pd
import json

a = "wine"
b = 1000
df = pd.read_csv(f"datasets/test/{a}.test.data")
df.iloc[:, -2].replace(2, int(0), inplace=True)
prop = {f"{a}": []}
for i in range(0, len(df), b):
    l = df.iloc[i:i+b, -2].tolist()
    size : int = len(l)
    pos_prop : float = round(sum(l) / size, 2)
    prop[f"{a}"].append(pos_prop)
    
with open(f"tables/{a}-prop.json", 'w') as f:
        json.dump(prop, f)