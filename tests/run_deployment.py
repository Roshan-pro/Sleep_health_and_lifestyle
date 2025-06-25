import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import click
from pipelines.deployment_pipeline import  continous_deployment_pipline,inference_pipeline
from rich import print
from zenml.integrations.mlflow.mlflow_utils import  get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service whwn done."
)

def run_main(stop_service:bool):
    model_name="Predicting_Sleep_Disorder"
    if stop_service:
        model_deployer=MLFlowModelDeployer.get_active_model_deployer()
        existing_services=model_deployer.find_model_server(
            pipeline_name="continous_deployment_pipline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=model_name,
            running=True
        )
        if existing_services:
            existing_services[0].stop(timeout=10)
        return
    continous_deployment_pipline()
    model_deployer=MLFlowModelDeployer.get_active_model_deployer()
    inference_pipeline()
    print(
        "Now run\n"
        f" mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To insert your experiment runs within the mlflow UI.\n"
        "Ypu can find your runs tracked within the mlflow_example_pipeline."
        "experiment.Here yo'll be be able o compare the two runs"
    )
    services=model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step"
        
    )
    if services[0]:
        print(f"""The Ml flow service is runing locally as daemon
              process and accepts inference request at:\n 
              {services[0].get_prediction_url}\n 
              to stop the service,re-run the same command and supply the
              '--stop-service' argument""")
if __name__ == "__main__":
    run_main()
#http://127.0.0.1:5000