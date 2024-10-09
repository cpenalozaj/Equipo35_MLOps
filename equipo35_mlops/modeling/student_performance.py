import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

from ..data_explorer import DataExplorer


class StudentPerformanceModel:
    def __init__(self):
        self.labels = ['Best', 'Good', 'Pass', 'Vg']
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.configured_models = {}

    def configure_model(self, model_name, params):
        name_to_model = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'xgboost': XGBClassifier
        }

        self.configured_models[model_name] = name_to_model[model_name](**params)

    @staticmethod
    def download_dataset(storage_path, file_name):
        student_data = fetch_ucirepo(name='Student Academics Performance')
        X = student_data.data.features
        y = student_data.data.targets
        # create dataset with features and target
        student_data_df = pd.concat([X, y], axis=1)
        student_data_df.head()

        file_path = os.path.join(storage_path, file_name)
        os.makedirs(storage_path, exist_ok=True)
        student_data_df.to_csv(file_path, index=False)

    @staticmethod
    def load_data(file_path, file_name):
        return pd.read_csv(os.path.join(file_path, file_name))

    @staticmethod
    def preprocessing_pipeline(categorical_features, numeric_features):
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return preprocessor

    def preprocess_data(self, file_path, file_name, output_path, model_path):
        # All students have unmarried status
        self.data = StudentPerformanceModel.load_data(file_path, file_name)
        student_data_df = self.data.drop('ms', axis=1)

        X = student_data_df.drop('esp', axis=1)
        y = pd.DataFrame(student_data_df['esp'])

        # Encode the target labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.values.ravel())  # Convert DataFrame to NumPy array and flatten

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        categorical_features = self.X_train.select_dtypes(include=['object']).columns
        numeric_features = self.X_train.select_dtypes(include=['number']).columns

        # fit transformers only on training data to avoid data leakage
        self.preprocessor = self.preprocessing_pipeline(categorical_features, numeric_features)

        X_train_preprocessed = self.preprocessor.fit_transform(self.X_train)
        # Aplicar SMOTE para balancear las clases
        smote = SMOTE()
        # oversampling training data
        self.X_train, self.y_train = smote.fit_resample(X_train_preprocessed, self.y_train)

        # store to fs for dvc pipeline

        np.save(os.path.join(output_path, 'X_train.npy'), self.X_train)
        np.save(os.path.join(output_path, 'y_train.npy'), self.y_train)
        np.save(os.path.join(output_path, 'X_test.npy'), self.X_test)
        np.save(os.path.join(output_path, 'y_test.npy'), self.y_test)

        # store preprocessor to fs for dvc pipeline
        joblib.dump(self.preprocessor, os.path.join(model_path, 'preprocessor.pkl'))

        return self

    def train_model(self, model_path, processed_data_path, model_params, model_name='logistic_regression'):
        # load stages data
        if not self.y_train or not self.X_train:
            self.X_train = np.load(os.path.join(processed_data_path, 'X_train.npy'), allow_pickle=True)
            self.y_train = np.load(os.path.join(processed_data_path, 'y_train.npy'), allow_pickle=True)

        self.configure_model(model_name, model_params[model_name])
        self.configured_models[model_name].fit(self.X_train, self.y_train)
        # joblib.dump(self.models[model_name], os.path.join(model_path, f'{model_name}.pkl'))
        return self

    def evaluate_model(self, model_name='logistic_regression'):
        # load preprocessor
        if not self.preprocessor:
            self.preprocessor = joblib.load(os.path.join(model_path, 'preprocessor.pkl'))

        X_test_preprocessed = self.preprocessor.transform(self.X_test)

        print("Model evaluation")
        y_pred = self.configured_models[model_name].predict(X_test_preprocessed)
        DataExplorer.plot_confusion_matrix(self.y_test, y_pred, self.labels)

        report = classification_report(self.y_test, y_pred)
        print("Classification Report:")
        print(report)

        # Accuracy Score
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
        return self


# def main():
#     model = StudentPerformanceModel()
#     model.load_data(
#         file_path='data/raw',
#         file_name='student_performance.csv'
#     ).preprocess_data().train_model().evaluate_model()
#
#
# if __name__ == "__main__":
#     main()
