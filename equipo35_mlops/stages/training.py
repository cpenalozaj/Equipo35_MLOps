import dvc.api
from loguru import logger

from equipo35_mlops.modeling.student_performance import StudentPerformanceModel

params = dvc.api.params_show()
studentPerformance = StudentPerformanceModel()

logger.info("Training model...")
studentPerformance.train_model(model_path=params['shared']['models_path'],
                               processed_data_path=params['shared']['processed_data_path'],
                               model_name=params['target_model'],
                               model_params=params['models']
                               )
logger.success("Model training complete.")