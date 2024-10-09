import dvc.api
from loguru import logger

from equipo35_mlops.modeling.student_performance import StudentPerformanceModel

params = dvc.api.params_show()
studentPerformance = StudentPerformanceModel()

logger.info("Evaluating model...")
studentPerformance.evaluate_model(model_name=params['target_model'])
logger.success("Model evaluation complete.")