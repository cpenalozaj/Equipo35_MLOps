import dvc.api
from loguru import logger

from equipo35_mlops.modeling.student_performance import StudentPerformanceModel

params = dvc.api.params_show()

logger.info("Downloading data...")
StudentPerformanceModel.download_dataset(
    storage_path=params['shared']['raw_data_path'],
    file_name=params['shared']['raw_file_name']
)
logger.success("Data download complete.")