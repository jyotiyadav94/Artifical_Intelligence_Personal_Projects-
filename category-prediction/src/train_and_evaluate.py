import os 
import sys
import json
import torch
import argparse
import warnings
import logging 
import pandas as pd
from torch import nn
from transformers import BertModel
from urllib.parse import urlparse
sys.path.append(os.path.abspath('src'))
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from get_data import read_params, print_tabulated_data,setup_logging
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
warnings.filterwarnings("ignore")

class CategoryClassificationDataset(Dataset):
    # Dataset class for category classification task using BERT.
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}

class BERTClassifier(nn.Module):
    # BERT based classifier.
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

def get_device():
    # Get the device for PyTorch operations.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, data_loader, optimizer, scheduler, device):
    # Train the BERT model.
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()


def load_data(data_file, id2label):
    texts = data_file[['product_name', 'product_brand']].apply(lambda x: ' product_brand: '.join(x.dropna()), axis=1).tolist()
    labels = [key for label in data_file['category'] for key, value in id2label.items() if value == label]
    return texts, labels

def evaluate(model, data_loader, device):
    # Evaluate the BERT model.
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    precision = precision_score(actual_labels, predictions, average='weighted')
    recall = recall_score(actual_labels, predictions, average='weighted')
    f1 = f1_score(actual_labels, predictions, average='weighted')
    accuracy = accuracy_score(actual_labels, predictions)
    logger.info(f"Evaluation Precision: {precision}, Recall: {recall}, F1-score: {f1}, Accuracy: {accuracy}")

    return precision, recall, f1, accuracy

def save_model(model, model_dir, pred_dir):
    # Save the trained model.
    for directory in [model_dir, pred_dir]:
        os.makedirs(directory, exist_ok=True)
    paths = [os.path.join(model_dir, "model.pth"), os.path.join(pred_dir, "model.pth")]
    for path in paths:
        torch.save(model.state_dict(), path)
    logger.info("Model saved successfully.")

def id2label_mapping(df, column_name):
    # Create a mapping from category IDs to labels.
    id2label = {i: label for i, label in enumerate(df[column_name].unique())}
    return id2label

def train_and_evaluate(config_path):
    # Main function to train and evaluate the BERT model.
    config = read_params(config_path)
    transform_data_path = config["transformed_data"]["final_dataframe"]
    split_ratio = config["model_parameters"]["test_size"]
    pretrained_model_name = config["model_parameters"]["bert_model_name"]
    max_length = config["model_parameters"]["max_length"]
    batch_size = config["model_parameters"]["batch_size"]
    num_epochs = config["model_parameters"]["epochs"]
    random_state = config["model_parameters"]["random_state"]
    shuffle = config["model_parameters"]["shuffle"]
    scores_file = config["reports"]["scores"]
    model_dir = config["model_dir"]
    pred_dir = config["pred_dir"]
    
    df = pd.read_csv(transform_data_path, sep=",")
    id2label = id2label_mapping(df, 'category')
    texts, labels = load_data(df, id2label)
    num_classes = len(labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=split_ratio, random_state=random_state)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    train_dataset = CategoryClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = CategoryClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    device = get_device()
    model = BERTClassifier(pretrained_model_name, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        logger.info("*********************** TRAINING ************************************")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        precision, recall, f1, accuracy = evaluate(model, val_dataloader, device)
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1: {f1:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")

    model.eval()
    with open(scores_file, "w") as f:
        scores = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy
        }
        json.dump(scores, f, indent=4)
    save_model(model, model_dir, pred_dir)

logger = setup_logging()
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
