import pandas as pd
import numpy as np
import os
import shutil

def make_tests(path_tests, dataset, positive_proportions) -> None:
    if not os.path.isdir(f"{path_tests}/{dataset}"):
        os.mkdir(f"{path_tests}/{dataset}")
    
    df = pd.read_csv(f"{path_tests}/{dataset}.test.csv")
    df['class'].replace(2, int(0), inplace=True)
    
    size = int(len(df)/4)

    context1 = df[df['context'] == 1]
    context2 = df[df['context'] == 2]

    for pos_prop1 in positive_proportions:
        for pos_prop2 in positive_proportions:
            c1_sample1, c1_sample2 = create_samples(context1, size, pos_prop1)
            
            c2_sample1, c2_sample2 = create_samples(context2, size, pos_prop2)
            
            test = pd.concat([c1_sample1, c2_sample1, c1_sample2, c2_sample2], ignore_index=True)
            test.to_csv(f"{path_tests}/{dataset}/{pos_prop1}_{pos_prop2}.csv", index=False)
    
    
    
    

def create_samples(df, size, pos_prop) -> list:
    
    positive_class = df[df['class'] == 1]
    negative_class = df[df['class'] == 0]

    n_positive = int(size * pos_prop)
    n_negative = size - n_positive

    # Shuffle both datasets
    df_positive_shuffled = positive_class.sample(frac=1)
    df_negative_shuffled = negative_class.sample(frac=1)

    # Select n instances from the positive class dataset
    positive_sample = df_positive_shuffled.iloc[:n_positive, :]
    rest_positive = df_positive_shuffled.iloc[n_positive:, :]

    # Select n instances from the negative class dataset
    negative_sample = df_negative_shuffled.iloc[:n_negative, :]
    rest_negative = df_negative_shuffled.iloc[n_negative:, :]

    # Concatenate positive and negative samples
    sample1 = pd.concat([positive_sample, negative_sample])
    sample2 = pd.concat([rest_positive, rest_negative])
    
    sample1 = sample1.sample(frac=1).reset_index(drop=True)
    sample2 = sample2.sample(frac=1).reset_index(drop=True)
    
    return sample1, sample2
