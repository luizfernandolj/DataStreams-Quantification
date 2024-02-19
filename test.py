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
    path = f"{os.getcwd()}/tables"
    path_test = f"{os.getcwd()}/datasets/test"
    path_train = f"{os.getcwd()}/datasets/training"

    files_test = os.listdir(path_test)
    datasets = [file.split(".")[0] for file in files_test]
    files_test = [pd.read_csv(f"{path_test}/{f}") for f in files_test]

    files_train = os.listdir(path_train)
    files_train = [pd.read_csv(f"{path_train}/{f}") for f in files_train]

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)

    table = pd.DataFrame()
    window_size = 11 #window parameter to build the images for comparison
    table

    vet_accs = {}
    for i, files in enumerate(zip(files_train, files_test)):
        
        # taking the context 
        contexts = files[1].iloc[:, -1].tolist()
        contexts = [1 if contexts[i-1] != x else 0 for i, x in enumerate(contexts) ]
        real_drifts = [i for i, x in enumerate(contexts) if x == 1]
        
        # creating train, test datastreams
        train = files[0].iloc[:, :-1]
        test = files[1].iloc[:, :-1]
        train.iloc[:, -1].replace(2, int(0), inplace=True)
        test.iloc[:, -1].replace(2, int(0), inplace=True)

        print(f"dataset: {datasets[i]}")
        vet_accs_table = pd.DataFrame()
        prop_win = {}
        
                #IBDD RUN
        epsilon = 3
        ibdd = IBDD(train, test, window_size, clf, epsilon)
        vet_accs, drift_points, window_proportions = ibdd.run_sliding_window()
        vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
        drift_points_ibdd = [1 if x in drift_points else 0 for x in range(len(contexts))]
        prop_win["IBDD"] = window_proportions

        #IKS RUN
        ca = 1.90
        iks = IKS(train, test, window_size, clf, ca)
        vet_accs, drift_points, window_proportions = iks.run_sliding_window()
        vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
        drift_points_iks = [1 if x in drift_points else 0 for x in range(len(contexts))]
        prop_win["IKS"] = window_proportions
        
        #WRS RUN
        threshold = 0.001
        wrs = WRS(train, test, window_size, clf, threshold)
        vet_accs, drift_points, window_proportions = wrs.run_sliding_window()
        vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
        drift_points_wrs = [1 if x in drift_points else 0 for x in range(len(contexts))]
        prop_win["WRS"] = window_proportions
        
        
        drift_table = pd.DataFrame({"IKS":drift_points_iks, "IBDD":drift_points_ibdd, "WRS":drift_points_wrs, "REAL":contexts})
        prop_win_table = pd.DataFrame(prop_win)
        vet_accs_table = pd.concat([vet_accs_table, test.iloc[:, -1]], ignore_index=True)
        
        # Saving the dataframes into files for each dataset
        prop_win_table.to_csv(f"{path}/{datasets[i]}-prop.csv", index=False)
        drift_table.to_csv(f"{path}/{datasets[i]}-drift.csv", index=False)
        vet_accs_table.to_csv(f"{path}/{datasets[i]}-pred.csv", index=False)
     
    
if __name__ == "__main__":
    print('Iniciando...\n')
    run()
    print('Fim')