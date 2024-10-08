import typer
from loguru import logger
from tqdm import tqdm
import dvc.api

# from equipo35_mlops.config import FIGURES_DIR, PROCESSED_DATA_DIR
from equipo35_mlops.modeling.student_performance import  StudentPerformanceModel

app = typer.Typer()
params = dvc.api.params_show()

studentPerformance = StudentPerformanceModel()
@app.command()
def download_dataset():
    logger.info("Downloading data...")
    studentPerformance.download_dataset(
        storage_path=params['shared']['rawdata_path'],
        file_name=params['shared']['raw_file_name']
    )
    logger.success("Data download complete.")

@app.command()
def preprocess():
    logger.info("Preprocessing data...")
    studentPerformance.preprocess_data(
        file_path=params['shared']['rawdata_path'],
        file_name=params['shared']['raw_file_name'],
        output_path=params['shared']['processed_data_path'],
        model_path=params['shared']['models_path']
    )
    logger.success("Data preprocessing complete.")

@app.command()
def training():
    logger.info("Training model...")
    studentPerformance.configure_models(params['models'])
    studentPerformance.train_model(model_path=params['shared']['models_path'],
                                   processed_data_path=params['shared']['processed_data_path'],
                                   model_name='logistic_regression',
                                   )
    logger.success("Model training complete.")

@app.command()
def evaluate():
    logger.info("Evaluating model...")
    studentPerformance.evaluate_model('logistic_regression')
    logger.success("Model evaluation complete.")
