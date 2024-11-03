import pytest
import dvc.api

from equipo35_mlops.modeling.student_performance import StudentPerformanceModel

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
def test_random_states_set_for_reproducibility(pipeline):
    """
    Tests the pipeline configuration.
    """
    for key, value in pipeline.configured_models.items():
        assert value.random_state == 42
    
def test_model_parameters_within_healthy_ranges(pipeline):
    """
    Tests that model parameters are within healthy ranges.
    """
    healthy_ranges = {
        'n_estimators': (50, 200),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 5)
    }

    for model_name, model in pipeline.configured_models.items():
        if hasattr(model, 'n_estimators'):
            assert healthy_ranges['n_estimators'][0] <= model.n_estimators <= healthy_ranges['n_estimators'][1]
        if hasattr(model, 'max_depth'):
            assert healthy_ranges['max_depth'][0] <= model.max_depth <= healthy_ranges['max_depth'][1]
        if hasattr(model, 'learning_rate'):
            assert healthy_ranges['learning_rate'][0] <= model.learning_rate <= healthy_ranges['learning_rate'][1]
        if hasattr(model, 'min_samples_split'):
            assert healthy_ranges['min_samples_split'][0] <= model.min_samples_split <= healthy_ranges['min_samples_split'][1]
        if hasattr(model, 'min_samples_leaf'):
            assert healthy_ranges['min_samples_leaf'][0] <= model.min_samples_leaf <= healthy_ranges['min_samples_leaf'][1]
