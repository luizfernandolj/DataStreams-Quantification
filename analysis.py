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

def plot_acc(vet_acc, window, marker_type, line, method_name):
  vet_len = len(vet_acc)
  mean_acc = []
  for i in range(0, vet_len, window):
      mean_acc.append(np.mean(vet_acc[i:i+window]))
  
  fig, ax = plt.subplots(figsize=(4, 2))
  plt.plot([float(x)*window for x in range(0,len(mean_acc))], mean_acc, marker=marker_type, ls=line, label=method_name )
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.xlabel('Examples')
  plt.ylabel('Accuracy') 
  plt.legend()


"""
def prediction_analysis(vet_pred):
    df = pd.read_csv(f"{path}/tables/{vet_pred}")
    true_labels = df.iloc[:, -1]
    df.drop("real", axis=1, inplace=True)
    
    vet_acc = {}
    for c in df.columns:
        mean_acc = []
        for i in range(0, len(df), window_size):      
            mean_acc.append(np.mean(df[c][i:i+window_size])) 
        vet_acc[c] = mean_acc
        #plt.plot([float(x)*window_size for x in range(0,len(vet_acc[c]))], vet_acc[c], label=c )
    #v
    print(pd.DataFrame(vet_acc))
    sns.boxenplot(data=pd.DataFrame(vet_acc))
    plt.xticks(rotation=45)
    plt.show()
"""

def prediction_analysis(vet_pred, window_size):
    df = pd.read_csv(f"{path}/tables/{vet_pred}")
    true_labels = df.iloc[:, -1].tolist()
    df.drop("real", axis=1, inplace=True)
    
    vet_acc = {}
    for c in df.columns:
        print(c)
        mean_acc = []
        for i in range(1, len(df)): 
            acc = round(accuracy_score(y_true = true_labels[0:i], y_pred = df[c].tolist()[0:i]), 2)      
            mean_acc.append(acc) 
        vet_acc[c] = mean_acc
        #plt.plot(range(1, len(df), 500), vet_acc[c], label=c )
    vet_acc = pd.DataFrame(vet_acc)
    print(vet_acc)
    #img = px.line(vet_acc)
    #fig = sns.lineplot(data=vet_acc, palette="tab10", dashes=False, markers=True)
    fig = sns.boxplot(vet_acc)
    #fig.set_ylim([0.5, 1])
    plt.show()
    #plt.ylim([0.5, 0.9])
    #plt.xlim([1, len(df)])
    #plt.legend()
    #plt.show()
    #plt.xticks(rotation=45)
    #img.show()
    
    
for i, dataset in enumerate(datasets):
    vet_drift, vet_pred, prop_win = list(filter(lambda x: dataset in x, f))
    
    print("Prediction Analysis...\n")
    prediction_analysis(vet_pred, window_size[i])