# My Personal Projects

Welcome to my GitHub repository containing various personal projects. Below is a list of each project along with a brief description and the technologies used.

## Project List

## 1. Project 1 - Agents
**Description:** The Agent was exposed to Rest API provides endpoints for interacting with a copywriting agent. It allows users to generate copy for various purposes such as menu items, social media posts, advertising content, and newsletters.

**Technologies Used:** angchain, Agents , LLM ,Chains,FastAPI, Docker, Python

**Project Structure:**
```bash
fastApi/
│
├── copywritingAgent/
│ ├── __init__.py
│ ├── router.py
│ ├── main.py
│ ├── .env
│
├── Dockerfile
├── README.md
├── docker-compose.yaml
└── requirements.txt
```


## 2. Project 2 - Insurance Claim 

Task Description: Examine the quality of the attached dataset. Use ML (Python) to find insights, hidden patterns, and forecast trends; anything related to financial and risk forecasting, as well as ideal customer profile (ICP).
**Instructions**: You may invest as much time and effort as you wish. You can complete the task at your discretion, using whichever methods, libraries, and tools you think are most effective.

Project Structure
In this Report, we will examine the Exploratory Data Analysis for Machine learning. For Modelling Part of Machine Learning & LLM Refer to the Notebooks.

```bash
InsuranceClaims/
│
├── Notebooks/
│   ├── MachineLearningEDA+Modelling.ipynb
│   └── LLM_fine_tuningipynb.ipynb
│
├── dataset/
│   ├── datasetqa.csv
│   ├── features.csv
│   └── formatted.json
│
├── images/
│   └── (all images used in README.md)
│
├── Reports/
│   ├── ReportPandasProfiling.html
│   ├── sweetviz_report.html
│
├── .gitignore
├── LICENSE
└── README.md
```
* It utilizes Machine learning Models Statistical Analysis , Data Preprocessing , Exploratory Data Analysis using Sweetviz ,Autoviz, Pandas profiling ,Encoding, Modelling using LazyPredict
* It utilizes LLM to predict the insurance Claim


## 3. Project 3 - Fine-Tuning LLM Model for Copywriting Agent

We are fine-tuning a Language Model (LLM) for various copywriting tasks, using different datasets and prompts. The tasks include generating restaurant menu descriptions, social media posts, advertising copy, and newsletter campaigns.
* It utilises LLM,Agents, reAct Framework , Langchain ,Tools,  ChatGroq



## 4. Project 4 - Sales Prediction Project

**Project Introduction**
This project focuses on predicting sales for thousands of product families available at Favorita stores in Ecuador. The dataset provided includes various features such as dates, store and product information, promotional status, and sales figures. The aim is to accurately forecast sales based on these attributes.

Main Goal
The primary objective of this project is to enhance the Kaggle score for store sales prediction in Ecuador. This will be achieved through the following comparison:

* Utilizing conventional machine learning models such as Linear Regression, Random Forest, and XGBoost.
* Fine-tuning an open-source Large Language Model (LLM) like FLANT-5.
* Deployment of Demo on HuggingFace


## 5. Project 5 - Wine Quality Prediction

Description
* This project predicts the quality of wine using machine learning techniques. 
* It utilizes MLFLOW, DVC, CI/CD, frameworks, Docker, and docker-compose.yaml for seamless development and deployment,MLOPS.
