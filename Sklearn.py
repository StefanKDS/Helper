# Load data
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

# Data infos
list(data.target_names)
list(data.feature_names)

import pandas as pd
# Read the DataFrame and show the first 5 rows
pd.DataFrame(data.data, columns=data.feature_names).head()

# Normalize features
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data.data)
df = pd.DataFrame(x_scaled, columns=data.feature_names)

# Do we miss some values ?
df.info()

# Split data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, data.target)
