import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier


class DataExplorer:

    @staticmethod
    def explore_data(data):
        '''
        Prints the first 5 rows, the data description and the data info.

        Parameters:
        data: The dataframe to explore

        Returns:
        None

        '''
        print(data.head().T)
        print(data.describe())
        print(data.info())

    @staticmethod
    def plot_histograms(dataframe, column_names):
        '''
        Plots histograms for the specified columns in the dataframe.

        Parameters:
        dataframe: The dataframe containing the data to plot
        column_names: A list of column names to plot

        Returns:
        None
        
        '''
        num_columns = len(column_names)
        num_rows = (num_columns + 1) // 2  # Calculate the number of rows needed
        sns.set(style="whitegrid")
        fig, axs = plt.subplots(num_rows, 3, figsize=(12, num_rows * 5))
        axs = axs.flatten()

        for i, columna in enumerate(column_names):
            # Get unique categories and assign a color for each one
            unique_values = dataframe[columna].unique()
            sns.countplot(x=columna, data=dataframe, ax=axs[i], hue=columna, dodge=False, palette="Set2", legend=False)
            axs[i].set_title(f'Gráfico de Barras de Recuento para {columna}')
            axs[i].set_xlabel(columna)
            axs[i].set_ylabel('Frecuencia')
            axs[i].tick_params(axis='x', rotation=45)

        # Remove any unused axes if necessary
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=['Best', 'Good', 'Pass', 'Vg']):
        """
        This function plots a confusion matrix using Seaborn's heatmap with colors, rectangles, and labels.

        Parameters:
        - y_true: The true labels
        - y_pred: The predicted labels
        - labels: A list of labels for the x and y axes (classes)

        Example usage:
        plot_confusion_matrix(y_test, y_pred, labels=['Best', 'Good', 'Pass', 'Vg'])
        """

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels, linewidths=1, linecolor='black')

        # Add labels, title, and axis titles
        plt.xlabel('Predicted Labels', fontsize=14)
        plt.ylabel('True Labels', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)

        # Display the plot
        plt.show()


class StudentPerformanceModel:

    def __init__(self, filepath='./data/raw'):
        self.filepath = filepath
        self.labels = ['Best', 'Good', 'Pass', 'Vg']
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest': RandomForestClassifier(class_weight='balanced'),
            'svm': SVC(kernel='linear', class_weight='balanced'),
            'xgboost': XGBClassifier(eval_metric='mlogloss')
        }

    def download_dataset(self):
        student_data = fetch_ucirepo(name='Student Academics Performance')
        X = student_data.data.features
        y = student_data.data.targets
        # create dataset with features and target
        student_data_df = pd.concat([X, y], axis=1)
        student_data_df.head()

        file_path = os.path.join(self.filepath, 'student_data_df.csv')
        os.makedirs(self.filepath, exist_ok=True)
        student_data_df.to_csv(file_path, index=False)

    def load_data(self, file_name="student_data_df.csv"):
        print("Loading data...")
        self.data = pd.read_csv(os.path.join(self.filepath, file_name))
        print("Data loaded")
        DataExplorer.explore_data(self.data)
        return self

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

    def preprocess_data(self):
        # All students have unmarried status
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

        return self

    def train_model(self, model_name='logistic_regression'):
        self.models[model_name].fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self, model_name='logistic_regression'):
        # use previously trained preprocessor on training data
        X_test_preprocessed = self.preprocessor.transform(self.X_test)

        print("Model evaluation")
        y_pred = self.models[model_name].predict(X_test_preprocessed)
        DataExplorer.plot_confusion_matrix(self.y_test, y_pred, self.labels)

        report = classification_report(self.y_test, y_pred)
        print("Classification Report:")
        print(report)

        # Accuracy Score
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
        return self


def main():
    model = StudentPerformanceModel()
    model.load_data().preprocess_data().train_model().evaluate_model()


if __name__ == "__main__":
    main()
