import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.metrics import accuracy_score
import json
import plotly.graph_objects as go

path = f"{os.getcwd()}"
datasets = [file.split(".")[0] for file in os.listdir(f"{path}/datasets/test")]
f = os.listdir(f"{path}/tables")
window_size = [1000, 1000, 200, 300]

def prediction_analysis(vet_pred, window_size):
    df = pd.read_csv(f"{path}/tables/{vet_pred}")
    true_labels = df.iloc[:, -1].tolist()
    df.drop("real", axis=1, inplace=True)
    df = df[['IBDD', 'IBDD-DyS', 'IKS', 'IKS-DyS', 'WRS', 'WRS-DyS']]
    len_df = len(df)
    
    vet_acc = {}
    for c in df.columns:
        mean_acc = []
        for i in range(0, len(df), int(window_size)): 
            acc = round(accuracy_score(y_true = true_labels[i:i+int(window_size)], y_pred = df[c].tolist()[i:i+int(window_size)]), 2)      
            mean_acc.append(acc) 
        vet_acc[c] = mean_acc
    vet_acc = pd.DataFrame(vet_acc)
    print(vet_acc)
    return vet_acc, len_df
    
def proportion_analysis(prop_win: dict):
    f = open(f'{path}/tables/{prop_win}')
    proportion_window = json.load(f)
    
    props = proportion_window.copy()
    for dataset, prop in proportion_window.items():
        props[dataset] = round(np.array(prop).mean(), 2)
    return list(proportion_window.values())[0]
        
def plot(vet_acc, prop, window_size, dataset):
    print(prop)
    palette = sns.color_palette("bright", 6)
    img = sns.lineplot(vet_acc, dashes=False, markers=True, palette=palette)
    img.set_xticklabels([int(x)*int(window_size/2) for x in range(1, len(vet_acc)*2)])
    img.set_xlabel("Instances of the stream")
    img.set_ylabel("Accuracy of the windows")
    img.set_title(f"{dataset}")
    plt.show()
    
    
    
for i, dataset in enumerate(datasets):
    vet_drift, vet_pred, prop_win = list(filter(lambda x: dataset in x, f))
    
    
    print("Proportion analysis...\n")
    props = proportion_analysis(prop_win)
    
    print("Prediction Analysis...\n")
    vet_acc, len_df = prediction_analysis(vet_pred, window_size[i])
    
    plot(vet_acc, props, window_size[i], dataset)