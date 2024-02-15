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
    window_size = 11#window parameter to build the images for comparison
    table

    vet_accs = {}
    for i, files in enumerate(zip(files_train, files_test)):
        contexts = files[1].iloc[:, -1]
        train = files[0].iloc[:, :-1]
        test = files[1].iloc[:, :-1]
        train.iloc[:, -1].replace(2, int(0), inplace=True)
        test.iloc[:, -1].replace(2, int(0), inplace=True)

        print(f"dataset: {datasets[i]}")
        row = pd.DataFrame()
        vet_accs_table = pd.DataFrame()
        row["dataset"] = [datasets[i]]


        threshold = 1.90
        iks = IKS(train, test, window_size, clf, threshold)
        vet_accs, drift_points, window_proportions = iks.run_sliding_window()
        vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
     
    
if __name__ == "__main__":
    print('Iniciando...\n')
    run()
    print('Fim')