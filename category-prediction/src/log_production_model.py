import os 
import sys
import json
import torch
import warnings
import logging 
import mlflow
import argparse
import pandas as pd
from torch import nn
from urllib.parse import urlparse
sys.path.append(os.path.abspath('src'))
from torch.utils.data import DataLoader, Dataset
from get_data import read_params, get_data, print_tabulated_data,setup_logging
from train_and_evaluate import read_params, CategoryClassificationDataset, get_device, train, load_data, evaluate, id2label_mapping, BERTClassifier
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

def log_mlflow_params(config):
    # Logs parameters to MLflow.
    for key, value in config.items():
        mlflow.log_param(key, value)

def mlflow_train_and_evaluate(config_path):
    # Trains and evaluates a BERT model using MLflow for tracking.
    logger.info("Starting MLflow run...")
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        log_mlflow_params(config)

        transform_data_path = config["transformed_data"]["final_dataframe"]
        split_ratio = config["model_parameters"]["test_size"]
        pretrained_model_name = config["model_parameters"]["bert_model_name"]
        max_length = config["model_parameters"]["max_length"]
        batch_size = config["model_parameters"]["batch_size"]
        num_epochs = config["model_parameters"]["epochs"]
        random_state = config["model_parameters"]["random_state"]
        shuffle = config["model_parameters"]["shuffle"]
        
        df = pd.read_csv(transform_data_path, sep=",")
        id2label=id2label_mapping(df,'category')
        texts, labels = load_data(df, id2label)
        num_classes=len(labels)
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=split_ratio, random_state=random_state)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        train_dataset = CategoryClassificationDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = CategoryClassificationDataset(val_texts, val_labels, tokenizer, max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BERTClassifier(pretrained_model_name, num_classes).to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            train(model, train_dataloader, optimizer, scheduler, device)
            precision, recall, f1, accuracy = evaluate(model, val_dataloader, device)
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1: {f1:.4f}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1", f1)
            mlflow.log_metric("Accuracy", accuracy)
        
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.pytorch.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.pytorch.save_model(model, "model")
            
        logger.info("MLflow run completed.")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    logger = setup_logging()
    mlflow_train_and_evaluate(config_path=parsed_args.config)
