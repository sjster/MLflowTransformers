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


def run_databricks():
	project_url = "./mlflow_project"
	experiment_name = "/Users/srijith.rajamohan@databricks.com/transformers"
	experiment_id = "12009229"
	params = {"data_file" : "./data/data.csv",
          "remote" : 1}
	backend = "databricks"
	config_file = "./cluster_config/create_cluster.json"
	mlflow_run_res = mlflow.projects.run(uri=project_url,
                                        entry_point="databricks",
					experiment_id=experiment_id,
					backend=backend,
					backend_config=config_file,
					parameters=params)
	print("Job submitted")
	mlflow_run_res.wait()
	return(mlflow_run_res.get_status())

def test_mlflow_runs():
	assert(run_databricks() == 'FINISHED')


