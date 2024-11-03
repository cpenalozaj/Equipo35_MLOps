
import pytest
import numpy as np
import dvc.api

from equipo35_mlops.modeling.student_performance import StudentPerformanceModel
from equipo35_mlops.schemas.data_schemas import student_data_schema

# fixtures
@pytest.fixture
def pipeline():
    params = dvc.api.params_show()
    studentPerformance = StudentPerformanceModel()
    studentPerformance.preprocess_data(
        file_path=params["shared"]["raw_data_path"],
        file_name=params["shared"]["raw_file_name"],
        output_path=params["shared"]["processed_data_path"],
        model_path=params["shared"]["models_path"],
    )
    return studentPerformance

# tests

def test_labels_are_encoded(pipeline):
    """
    Tests the labels are encoded as integers.
    """
    assert pipeline.y_train.dtype == 'int64'
        

def test_encoded_labels_match_labels(pipeline):
    """
    Tests the label encodings match the labels quantity. 
    """
    assert len(np.unique(pipeline.y_train)) == len(pipeline.labels)
        
def test_no_missing_data_after_preprocessing(pipeline):
    """
    Tests there are no missing values in the data after preprocessing.
    """
    assert not np.isnan(pipeline.X_train_preprocessed.todense()).any()

def test_features_as_numeric_after_preprocessing(pipeline):
    """
    Tests all features are numeric after preprocessing.
    """
    # Convert to dense matrix if it's a sparse matrix
    X_train_dense = pipeline.X_train_preprocessed.todense() if hasattr(pipeline.X_train_preprocessed, 'todense') else pipeline.X_train_preprocessed
    
    # Check if all elements are of a numeric type
    assert np.issubdtype(X_train_dense.dtype, np.number)