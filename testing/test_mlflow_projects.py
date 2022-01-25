import mlflow
import pytest

def run_local():
	project_url = "./mlflow_project"
	experiment_name = "transformers"
	params = {"data_file" : "./data/data.csv",
          "remote" : 0}
	mlflow_run_res = mlflow.projects.run(uri=project_url, experiment_name=experiment_name, parameters=params)
	print("Job submitted")
	mlflow_run_res.wait()
	return(mlflow_run_res.get_status())


def test_mlflow_runs():
	assert(run_local() == 'FINISHED')


