import mlflow
import mlflow.sklearn
from loguru import logger
import dvc.api


from equipo35_mlops.modeling.student_performance import StudentPerformanceModel
dvc_params = dvc.api.params_show()

studentPerformance = StudentPerformanceModel()

studentPerformance.preprocess_data(
    file_path=dvc_params['shared']['raw_data_path'],
    file_name=dvc_params['shared']['raw_file_name'],
    output_path=dvc_params['shared']['processed_data_path'],
    model_path=dvc_params['shared']['models_path']
)

def run_experiment(experiment_name):
    mlflow.set_tracking_uri(dvc_params['mlflow_server_uri'])
    mlflow.set_experiment(f"/{experiment_name}")
    for model_name, model_params in dvc_params['models'].items():
        with mlflow.start_run(run_name=model_name):
            logger.info("Training model...")
            studentPerformance.train_model(model_path=dvc_params['shared']['models_path'],
                                           processed_data_path=dvc_params['shared']['processed_data_path'],
                                           model_name=model_name,
                                           model_params=model_params
                                           )

            logger.info("Evaluating model...")
            eval_metrics = studentPerformance.evaluate_model(
                model_name=model_name,
                model_path=dvc_params['shared']['models_path'],
                processed_data_path=dvc_params['shared']['processed_data_path']
            )

            # log params and metrics to mlflow
            mlflow.log_params(model_params)
            mlflow.log_metrics({
                'accuracy': eval_metrics['accuracy'],
                'precision': eval_metrics['precision'],
                'recall': eval_metrics['recall']
            })
            mlflow.sklearn.log_model(studentPerformance.configured_models[model_name], artifact_path="models")

            logger.success("Model evaluation complete.")

run_experiment("student_performance")