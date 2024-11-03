import pytest
import dvc.api
from sklearn.metrics import accuracy_score, precision_score

from equipo35_mlops.modeling.student_performance import StudentPerformanceModel

# fixtures
@pytest.fixture
def pipelines():
    params = dvc.api.params_show()
    pipeline1 = StudentPerformanceModel()
    pipeline1.evaluate_model(
        model_name=params['target_model'],
        model_path=params['shared']['models_path'],
        processed_data_path=params['shared']['processed_data_path']
    )
    pipeline2 = StudentPerformanceModel()
    pipeline2.evaluate_model(
        model_name='random_forest',
        model_path=params['shared']['models_path'],
        processed_data_path=params['shared']['processed_data_path']
    )
    return pipeline1, pipeline2

# tests

def test_accuracy_higher_than_benchmark(pipelines):
    pipeline1, _ = pipelines
    
    actual_accuracy = pipeline1.get_accuracy()
    
    benchmark_predictions = [1.0] * len(pipeline1.y_test)
    benchmark_accuracy = accuracy_score(y_true=pipeline1.y_test, y_pred=benchmark_predictions)
    
    # Check if the model accuracy is higher than the benchmark accuracy
    assert actual_accuracy > benchmark_accuracy

def test_precision_higher_than_benchmark(pipelines):
    pipeline1, _ = pipelines
    
    actual_precision = pipeline1.get_precision()
    
    benchmark_predictions = [1.0] * len(pipeline1.y_test)
    benchmark_precision = precision_score(y_true=pipeline1.y_test, y_pred=benchmark_predictions, average='weighted', zero_division=0.0)
    
    # Check if the model precision is higher than the benchmark precision
    assert actual_precision > benchmark_precision

    
def test_accuracy_compared_to_previous_version(pipelines):
    pipeline1, pipeline2 = pipelines
    
    accuracy_v1 = pipeline1.get_accuracy()
    accuracy_v2 = pipeline2.get_accuracy()
    
    print(f'Accuracy of model 1: {accuracy_v1}')
    print(f'Accuracy of model 2: {accuracy_v2}')
    
    # Comparing the accuracy of the second model against the first one
    assert accuracy_v2 >= accuracy_v1