import os
import shutil
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from Experiment import Experiment
from detectors.IBDD import IBDD
from detectors.IKS import IKS
from detectors.WRS import WRS
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def run(dataset, window_size):
    path = f"{os.getcwd()}/tables"
    path_test = f"{os.getcwd()}/datasets/test/{dataset}"
    path_train = f"{os.getcwd()}/datasets/training/{dataset}"
    
    ibdd_dir = f"{os.getcwd()}/ibdd_folder/{dataset}"
    if not os.path.isdir("ibdd_folder"):
        os.mkdir(f"{os.getcwd()}/ibdd_folder")
    if os.path.isdir(ibdd_dir):
        shutil.rmtree(ibdd_dir)
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
    
    threshold = 0.001
    wrs = WRS(train, window_size, threshold)
    exp = Experiment(train, test, window_size, clf, wrs, 'WRS')
    vet_accs, drift_points = exp.run_stream()
    vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
    drift_points_wrs = [1 if x in list(drift_points.values())[0] else 0 for x in range(len(contexts))]
    
    
    
    ca = 1.95
    iks = IKS(train, window_size, ca)
    exp = Experiment(train, test, window_size, clf, iks, 'IKS')
    vet_accs, drift_points = exp.run_stream()
    vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
    drift_points_iks = [1 if x in list(drift_points.values())[0] else 0 for x in range(len(contexts))]
    
    #IBDD RUN
    epsilon = 3
    ibdd = IBDD(train.iloc[:, :-1], epsilon, window_size, dataset)
    exp = Experiment(train, test, window_size, clf, ibdd, 'IBDD')
    vet_accs, drift_points = exp.run_stream()
    vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
    drift_points_ibdd = [1 if x in list(drift_points.values())[0] else 0 for x in range(len(contexts))]
    
    
    drift_table = pd.DataFrame({"IKS": drift_points_iks, "IBDD":drift_points_ibdd, "WRS":drift_points_wrs, "REAL":contexts})
    vet_accs_table["real"] = test.iloc[:, -1].tolist()
    
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