import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from detectors.IKS import IKS
from detectors.IBDD import IBDD
from detectors.WRS import WRS
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def run(dataset, window_size):
    path = f"{os.getcwd()}/tables"
    path_test = f"{os.getcwd()}/datasets/test/{dataset}"
    path_train = f"{os.getcwd()}/datasets/training/{dataset}"
    
    ibdd_dir = f"{os.getcwd()}/ibdd_files/{dataset}"
      
    os.mkdir(ibdd_dir)

    test = pd.read_csv(f"{path_test}.test.csv")
    train = pd.read_csv(f"{path_train}.train.data")

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    
    
    # taking the context 
    contexts = test.iloc[:, -1].tolist()
    contexts = [1 if contexts[i-1] != x else 0 for i, x in enumerate(contexts) ]
    real_drifts = [i for i, x in enumerate(contexts) if x == 1]
    
    # creating train, test datastreams
    train = train.iloc[:, :-1].copy()
    test = test.iloc[:, :-1].copy()
    train.iloc[:, -1].replace(2, int(0), inplace=True)
    test.iloc[:, -1].replace(2, int(0), inplace=True)

    print(f"dataset: {dataset}")
    vet_accs_table = pd.DataFrame()
    prop_win = {}
    
    
    
    #IBDD RUN
    epsilon = 3
    ibdd = IBDD(train, test, window_size, clf, epsilon, ibdd_dir)
    vet_accs, drift_points, window_proportions = ibdd.run_sliding_window()
    vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
    drift_points_ibdd = [1 if x in list(drift_points.values())[0] else 0 for x in range(len(contexts))]
    
    prop_win["IBDD"] = window_proportions

    #IKS RUN
    ca = 1.90
    iks = IKS(train, test, window_size, clf, ca)
    vet_accs, drift_points, window_proportions = iks.run_sliding_window()
    vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
    drift_points_iks = [1 if x in list(drift_points.values())[0] else 0 for x in range(len(contexts))]
    prop_win["IKS"] = window_proportions
    
    #WRS RUN
    threshold = 0.001
    wrs = WRS(train, test, window_size, clf, threshold)
    vet_accs, drift_points, window_proportions = wrs.run_sliding_window()
    vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
    drift_points_wrs = [1 if x in list(drift_points.values())[0] else 0 for x in range(len(contexts))]
    prop_win["WRS"] = window_proportions
    
    
    drift_table = pd.DataFrame({"IKS":drift_points_iks, "IBDD":drift_points_ibdd, "WRS":drift_points_wrs, "REAL":contexts})
    vet_accs_table["real"] = test.iloc[:, -1].tolist()
    
    
    # Saving te dictionary into a new file
    with open(f"{path}/{dataset}-prop.json", 'w') as f:
        json.dump(prop_win, f)
    
    # Saving the dataframes into files for each dataset
    drift_table.to_csv(f"{path}/{dataset}-drift.csv", index=False)
    vet_accs_table.to_csv(f"{path}/{dataset}-pred.csv", index=False)
     
    
if __name__ == "__main__":
    print('Iniciando...\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('window', type=int)
    args = parser.parse_args()
    run(dataset=args.dataset, window_size=args.window)
    print('Fim')