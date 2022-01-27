import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import mlflow
import torch.nn as nn
import os
import sys


test_data_file = sys.argv[1]
remote = int(sys.argv[2])
print(test_data_file)
print(remote)

test_data_read = pd.read_csv(test_data_file)
test_data_read.dropna(inplace=True)

if(remote == 1):
  experiment_name = '/Users/srijith.rajamohan@databricks.com/transformers'
#else:
  #experiment_name = 'transformers'

#try:
#        mlflow.create_experiment(experiment_name)
#        mlflow.set_experiment(experiment_name)
#except:
#        mlflow.set_experiment(experiment_name)

#mlflow.autolog(log_input_examples=True, log_models=True, exclusive=False)

with mlflow.start_run() as mlflow_run:

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("johngiorgi/declutr-base")
    model = AutoModel.from_pretrained("johngiorgi/declutr-base")

    # Prepare some text to embed
    text = list(test_data_read['name'])
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Embed the text
    with torch.no_grad():
        sequence_output = model(**inputs)[0]

    # Mean pool the token-level embeddings to get sentence-level embeddings
    embeddings = torch.sum(
        sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
    ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)

    #mlflow.pyfunc.log_model(python_model=model, artifact_path="artifacts", registered_model_name="transformers_declutr")
    #mlflow.log_artifact(local_dir='/Users/srijith.rajamohan@databricks.com/transformers')

    # Compute a semantic similarity via the cosine distance
    semantic_sim = 1 - cosine(embeddings[0], embeddings[1])
    print(semantic_sim)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    #mlflow.log_param("threshold", 0.7)
    threshold = 0.7
    distance = cos(embeddings[0].repeat(len(embeddings),1), embeddings)
    print(distance)
    distance_match_index = torch.where(distance > threshold, True, False)
    print(distance_match_index)
    matched_text = test_data_read.loc[list(distance_match_index.numpy())]
    print(matched_text)

    if(not os.path.exists('./outputs')):
        os.mkdir('./outputs')

    test_data_read.to_csv('./outputs/text_data.csv')
    torch.save(distance, './outputs/distance.pt')
    matched_text.to_csv('./outputs/matched_text.csv')

    mlflow.log_artifacts('./outputs')
