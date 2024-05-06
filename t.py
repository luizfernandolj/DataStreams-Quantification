import pandas as pd
import json

a = "AedesQuinx"
df = pd.read_csv(f"datasets/test/{a}.test.csv")

df['class'].replace(2, int(0), inplace=True)
size = (len(df)/4)

context1 = df[df['context'] == 1]

positive_proportion = 0.5  # 20% of the minority class

# Get indices of the minority class
positive_class = context1[context1['class'] == 1]
negative_class = context1[context1['class'] == 0]

n_positive = int(size * positive_proportion)
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
new_dataset = pd.concat([positive_sample, negative_sample])

new_dataset = new_dataset.sample(frac=1).reset_index(drop=True)


print(new_dataset['class'].value_counts())
print()