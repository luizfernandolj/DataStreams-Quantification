import os
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Experiment import Experiment
from detectors.IBDD import IBDD
from detectors.IKS import IKS
from detectors.WRS import WRS
from utils.make_tests_imbalanced import make_tests
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def run(dataset, window_size, score_lenght, path_train, path_tests, path_results, classifier, positive_proportions):
    
    # CHECKING IF THE IBDD FOLDER ALREADY EXISTS, IF YES, CLEAR IT
    ibdd_dir = f"{os.getcwd()}/ibdd_folder/{dataset}"
    if not os.path.isdir("ibdd_folder"):
        os.mkdir(f"{os.getcwd()}/ibdd_folder")
    if os.path.isdir(ibdd_dir):
        shutil.rmtree(ibdd_dir)
    os.mkdir(ibdd_dir)
    
    
    # CREATING TEST FILES VARYING CLASS DISTRIBUTION IN EACH CONTEXT
    make_tests(path_tests, dataset, positive_proportions)
    

    # PASSING BY EACH TEST FILE
    for f in list(os.listdir(f"{path_tests}/{dataset}")):
        print(f)
        
        test = pd.read_csv(f"{path_tests}/{dataset}/{f}")
        train = pd.read_csv(f"{path_train}")
        train['class'].replace(2, int(0), inplace=True)

        # TAKING THE CONTEXT
        contexts = test.iloc[:, -1].tolist()
        contexts = [1 if contexts[i-1] != x else 0 for i, x in enumerate(contexts) ]
        real_drifts = [i for i, x in enumerate(contexts) if x == 1]
        
        # CREATING TRAIN, TEST DATASTREAMS
        train = train.iloc[:, :-1].copy()
        test = test.iloc[:, :-1].copy()

        print(f"dataset: {dataset}")
        vet_accs_table = pd.DataFrame()
        
        
        #IBDD RUN EXPERIMENT
        epsilon = 3
        ibdd = IBDD(train.iloc[:, :-1], epsilon, window_size, dataset)
        exp = Experiment(train, test, window_size, classifier, ibdd, 'IBDD', score_lenght)
        vet_accs_table, drift_points_ibdd = make_experiment(exp, contexts, vet_accs_table)
        
        
        #BASELINE RUN EXPERIMEN
        threshold = 0.001
        wrs = WRS(train, window_size, threshold)
        exp = Experiment(train, test, window_size, classifier, wrs, 'baseline', score_lenght)
        vet_accs_table, drift_points_baseline = make_experiment(exp, contexts, vet_accs_table)
        
        
        #IKS RUN EXPERIMEN
        ca = 1.95
        iks = IKS(train, window_size, ca)
        exp = Experiment(train, test, window_size, classifier, iks, 'IKS', score_lenght)
        vet_accs_table, drift_points_iks = make_experiment(exp, contexts, vet_accs_table)

        
        # CREATING VARIABLES TO SAVE TO A EXTERNAL FILE
        drift_table = pd.DataFrame({"IKS": drift_points_iks, "IBDD":drift_points_ibdd, "baseline":drift_points_baseline, "REAL":contexts})
        vet_accs_table["real"] = test.iloc[:, -1].tolist()
        
        
        # SAVING THE DATAFRAMES INTO FILES FOR EACH DATASET
        drift_table.to_csv(f"{path_results}/{f[:f.rfind('.')]}_{window_size}_{score_lenght}_drift.csv", index=False)
        vet_accs_table.to_csv(f"{path_results}/{f[:f.rfind('.')]}_{window_size}_{score_lenght}_pred.csv", index=False)
     
    

def make_experiment(experiment_object, contexts, vet_accs_table) -> list:
    """create a new experiment and saving the results into dataframes

    Args:
        experiment_object (Any): drift detector to be running the experiments on
        contexts (list): list of context
        vet_accs_table (DataFrame): pandas dataframe to save the results of experiment

    Returns:
        list: results of accuracies of predictions and the list of drift predictions
    """
    vet_accs, drift_points = experiment_object.run_stream()
    vet_accs_table = pd.concat([vet_accs_table, vet_accs], axis=1)
    drift_points = [1 if x in list(drift_points.values())[0] else 0 for x in range(len(contexts))]
    
    return vet_accs_table, drift_points

    
    
if __name__ == "__main__":
    print('Iniciando...\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('window', type=int)
    parser.add_argument('score_lenght', type=int)
    args = parser.parse_args()
    
    path_train = f"{os.getcwd()}/datasets/training/{args.dataset}.train.data"
    path_tests = f"{os.getcwd()}/datasets/test"
    path_results = f"{os.getcwd()}/results/"
    
    classifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    
    positive_proportions = [0.2, 0.5, 0.8]
    
    run(dataset=args.dataset, 
        window_size=args.window, 
        score_lenght=args.score_lenght,
        path_train=path_train, 
        path_tests=path_tests, 
        path_results=path_results,
        classifier=classifier,
        positive_proportions=positive_proportions)
    
    print('Fim')