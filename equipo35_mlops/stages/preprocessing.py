import dvc.api
from loguru import logger

from equipo35_mlops.modeling.student_performance import StudentPerformanceModel

params = dvc.api.params_show()
studentPerformance = StudentPerformanceModel()

logger.info("Preprocessing data...")
studentPerformance.preprocess_data(
    file_path=params['shared']['raw_data_path'],
    file_name=params['shared']['raw_file_name'],
    output_path=params['shared']['processed_data_path'],
    model_path=params['shared']['models_path']
)
logger.success("Data preprocessing complete.")