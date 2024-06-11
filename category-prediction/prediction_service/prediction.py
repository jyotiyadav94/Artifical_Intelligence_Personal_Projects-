import yaml
import os
import sys
import torch
import json
import joblib
import numpy as np
import pandas as pd
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
sys.path.append(os.path.abspath('category-prediction'))
from src.train_and_evaluate import get_device,BERTClassifier,id2label_mapping

params_path = "params.yaml"

def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict_category(text):
    config = read_params(params_path)
    bert_model_name = config["model_parameters"]["bert_model_name"]
    max_length = config["model_parameters"]["max_length"]
    model_path = config["webapp_model_dir"]
    num_labels = config["num_labels"]
    id2label = config["id2label"]
    model = BERTClassifier(bert_model_name, num_labels)
    device = get_device()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        
        # Return the label corresponding to the predicted numerical ID from id2label
        return id2label[preds.item()]
    
def preprocess(dict_request):
    processed_dict = {}
    for key, value in dict_request.items():
        processed_dict[key] = value.strip().lower()
    processed_string = ' '.join([f"{key}={value}" for key, value in processed_dict.items()])
    return processed_string

def form_response(dict_request):
    # Preprocess the input data
    input_data = preprocess(dict_request)
    print(input_data)
    # Get the prediction
    response = predict_category(input_data)
    return response

def api_response(dict_request):
    try:
        # Preprocess the input data
        input_data = preprocess(dict_request)
        
        # Get the prediction
        response = predict_category(input_data)

    except Exception as e:
        return {"error": str(e)}

