import csv
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_network import NeuralNet

headers = [
    "age",
    "sex",
    "chest_pain",
    "resting_blood_pressure",
    "serum_cholestoral",
    "fasting_blood_sugar",
    "resting_ecg_results",
    "max_heart_rate_achieved",
    "exercise_induced_angina",
    "oldpeak",
    "slope of the peak",
    "num_of_major_vessels",
    "thal",
    "heart_disease",
]
# reading the data file
heart_df = pd.read_csv("heart.csv", sep=" ", names=headers)

# dropping the targt column in training data
X = heart_df.drop(columns=["heart_disease"])

# replacing the target class with 0 and 1
# 1 means 'having heart diseaes' and 0 means 'not having heart diseases

heart_df["heart_disease"] = heart_df["heart_disease"].replace(1, 0)
heart_df["heart_disease"] = heart_df["heart_disease"].replace(2, 1)

y_label = heart_df["heart_disease"].values.reshape(X.shape[0], 1)

# split data into train ans test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_label, test_size=0.2, random_state=2
)

# standardize the dataset
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

print(f"Shape of train set is {X_train.shape}")
print(f"Shape of test set is {X_test.shape}")
print(f"Shape of train label is {y_train.shape}")
print(f"Shape of test labels is {y_test.shape}")


nn = NeuralNet()
nn.fit(X_train, y_train)
nn.plot_loss()
