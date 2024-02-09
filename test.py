import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from detectors.IKS import IKS
from detectors.IBDD import IBDD
from detectors.WRS import WRS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def run():
    #path = "C:/Users/luiz_/Projects/DataStreams/DataStreams-Quantification/Tables"
    #path_test = "C:/Users/luiz_/Projects/DataStreams/DataStreams-Quantification/datasets/test"
    #path_train = "C:/Users/luiz_/Projects/DataStreams/DataStreams-Quantification/datasets/training"
    path = f"{os.getcwd()}/Tables"
    path_test = f"{os.getcwd()}/datasets/test"
    path_train = f"{os.getcwd()}/datasets/training"

    files_test = os.listdir(path_test)
    datasets = [file.split(".")[0] for file in files_test]
    files_test = [pd.read_csv(f"{path_test}/{f}") for f in files_test]

    files_train = os.listdir(path_train)
    files_train = [pd.read_csv(f"{path_train}/{f}") for f in files_train]

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)

    table = pd.DataFrame()
    window_size = 300#window parameter to build the images for comparison
    t = 2000
    table

    vet_accs = {}
    for i, files in enumerate(zip(files_train, files_test)):
        contexts = files[1].iloc[:, -1]
        stream = pd.concat([files[0], files[1]], ignore_index=True)
        stream = stream.iloc[:, :-1]
        stream.iloc[:, -1].replace(2, int(0), inplace=True)

        print(f"dataset: {datasets[i]}")
        row = pd.DataFrame()
        vet_accs_table = pd.DataFrame()
        row["dataset"] = [datasets[i]]
        
        
        epsilon = 3
        ibdd = IBDD(stream, t, window_size, clf, contexts, epsilon)
        accs1, final_accs1 = ibdd.runslidingwindow()
        row = pd.concat([row, pd.DataFrame([accs1.mean().to_dict()])], axis=1)
        vet_accs_table = pd.concat([vet_accs_table, final_accs1], axis=1)


        threshold = 1.90
        iks = IKS(stream, t, window_size, clf, contexts, threshold)
        accs2, final_accs2 = iks.runslidingwindow()
        row = pd.concat([row, pd.DataFrame([accs2.mean().to_dict()])], axis=1)
        vet_accs_table = pd.concat([vet_accs_table, final_accs2], axis=1)


        threshold = 0.001
        wrs = WRS(stream, t, window_size, clf, contexts, threshold)
        accs3, final_accs3 = wrs.runslidingwindow()
        row = pd.concat([row, pd.DataFrame([accs3.mean().to_dict()])], axis=1)
        vet_accs_table = pd.concat([vet_accs_table, final_accs3], axis=1)

        vet_accs_table.to_csv(f"{path}/{datasets[i]}")
        table = pd.concat([table, row], ignore_index=True)
        table.to_csv(f"{path}/Table_accs")
     
    
if __name__ == "__main__":
    print('Iniciando...\n')
    run()
    print('Fim')