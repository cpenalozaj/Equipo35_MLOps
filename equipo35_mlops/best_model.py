import mlflow
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score

mlflow.set_tracking_uri('https://mlflow.dantumlogic.com')
run_id = "b14e873390af4e919b5846838e4d440d"
logged_model = f'runs:/{run_id}/models'

# Load model as a sklearn model.
loaded_model = mlflow.sklearn.load_model(logged_model)

X_test = pd.read_csv(os.path.join('data/processed', 'X_test.csv')) # test data is not preprocessed, just splitted
y_test = np.load(os.path.join('data/processed', 'y_test.npy'), allow_pickle=True)
y_pred = loaded_model.predict(X_test) # data is preprocessed inside the model

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")