import pytest
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

def test_input_data_values(pipeline):
    """
    Tests the input data complies with the schema values.
    """
    for column_name in pipeline.data.columns:
        assert set(pipeline.data[column_name].unique()) == set(student_data_schema[column_name]['values'])
        
def test_input_data_types(pipeline):
    """
    Tests the input data complies with the schema types.
    """
    for column_name in pipeline.data.columns:
        # make sure the column data type is the same as the schema data type
        assert pipeline.data[column_name].dtype == student_data_schema[column_name]['dtype']