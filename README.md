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


```bash
📦 
├─ Mistral_Fine_tuning.ipynb
├─ README.md
├─ agent.ipynb
├─ dataset
│  ├─ MenuDataset.csv
│  ├─ advertisingPrompt.csv
│  └─ social_media_prompt.csv
├─ inference_(1).ipynb
└─ mergeloadAdopterWithBase.ipynb
```


## 4. Project 4 - Sales Prediction Project

**Project Introduction**
This project focuses on predicting sales for thousands of product families available at Favorita stores in Ecuador. The dataset provided includes various features such as dates, store and product information, promotional status, and sales figures. The aim is to accurately forecast sales based on these attributes.

Main Goal
The primary objective of this project is to enhance the Kaggle score for store sales prediction in Ecuador. This will be achieved through the following comparison:

* Utilizing conventional machine learning models such as Linear Regression, Random Forest, and XGBoost.
* Fine-tuning an open-source Large Language Model (LLM) like FLANT-5.
* Deployment of Demo on HuggingFace

```bash
📦 
Images
│  └─ Screenshot 2024-03-22 at 21.13.38.png
LLM_fine_tuning
│  ├─ LLM_fine_tuningipynb.ipynb
dataset.py
datasetqatest.csv
submission.csv
Machine_Learning.ipynb
├─ README.md
├─ Report
│  ├─ report.html
│  └─ sweetviz_report.html
├─ model.pkl
├─ requirements.txt
├─ sample_submission.csv
└─ timeseries Research papers LLM
   ├─ FLANT-5.pdf
   ├─ Lag-Llama.pdf
   ├─ MOMENT.pdf
   ├─ Time-LLM.pdf
   └─ TimeGPT-1.pdf…
```


## 5. Project 5 - Wine Quality Prediction

Description
* This project predicts the quality of wine using machine learning techniques. 
* It utilizes MLFLOW, DVC, CI/CD, frameworks, Docker, and docker-compose.yaml for seamless development and deployment,MLOPS.


```bash
📦 
.dvc
│  ├─ .gitignore
│  └─ config
├─ .dvcignore
├─ .github
│  └─ workflows
│     └─ ci-cd.yaml
.gitignore
Dockerfile
README.md
├─ __pycache__
app.cpython-37.pyc
app.py
artifacts
│  └─ 1
│     ├─ 303917fcbbe1456d9862568bb2593f03
│     │  └─ artifacts
│     │     └─ model
│     │        ├─ MLmodel
│     │        ├─ conda.yaml
│     │        ├─ model.pkl
│     │        ├─ python_env.yaml
│     │        └─ requirements.txt
│     ├─ 40b53c2b198c45838f9a270489f7c046
│     │  └─ artifacts
│     │     └─ model
│     │        ├─ MLmodel
│     │        ├─ conda.yaml
│     │        ├─ model.pkl
│     │        ├─ python_env.yaml
│     │        └─ requirements.txt
│     ├─ a10af37f718647e8ac22a5eed6178bf7
│     │  └─ artifacts
│     │     └─ model
│     │        ├─ MLmodel
│     │        ├─ conda.yaml
│     │        ├─ model.pkl
│     │        ├─ python_env.yaml
│     │        └─ requirements.txt
│     ├─ a676bb4a4b004fe28f6a6e01f561e809
│     │  └─ artifacts
│     │     └─ model
│     │        ├─ MLmodel
│     │        ├─ conda.yaml
│     │        ├─ model.pkl
│     │        ├─ python_env.yaml
│     │        └─ requirements.txt
│     ├─ a6b36cb974054179b1c95c4c66a761ac
│     │  └─ artifacts
│     │     └─ model
│     │        ├─ MLmodel
│     │        ├─ conda.yaml
│     │        ├─ model.pkl
│     │        ├─ python_env.yaml
│     │        └─ requirements.txt
│     └─ dcb28f08c8d9478aaf450e92f886ffeb
│        └─ artifacts
│           └─ model
│              ├─ MLmodel
│              ├─ conda.yaml
│              ├─ model.pkl
│              ├─ python_env.yaml
│              └─ requirements.txt
├─ data
│  ├─ processed
│  │  └─ .gitignore
│  └─ raw
│     └─ .gitignore
├─ data_given
│  ├─ .gitignore
│  └─ winequality.csv.dvc
├─ docker-compose.yml
├─ dvc.lock
├─ dvc.yaml
├─ mlflow.db
├─ params.yaml
├─ prediction_service
│  ├─ __init__.py
│  ├─ __pycache__
│  │  ├─ __init__.cpython-37.pyc
│  │  └─ prediction.cpython-37.pyc
│  ├─ model
│  │  └─ model.joblib
│  ├─ prediction.py
│  └─ schema_in.json
├─ requirements.txt
├─ setup.py
├─ src.egg-info
│  ├─ PKG-INFO
│  ├─ SOURCES.txt
│  ├─ dependency_links.txt
│  └─ top_level.txt
├─ src
│  ├─ __init__.py
│  ├─ __pycache__
│  │  ├─ __init__.cpython-37.pyc
│  │  └─ get_data.cpython-37.pyc
│  ├─ get_data.py
│  ├─ load_data.py
│  ├─ log_production_model.py
│  ├─ split_data.py
│  └─ train_and_evaluate.py
├─ tests
│  ├─ __init__.py
│  ├─ __pycache__
│  │  ├─ __init__.cpython-37.pyc
│  │  ├─ conftest.cpython-37-pytest-7.4.4.pyc
│  │  └─ test_config.cpython-37-pytest-7.4.4.pyc
│  ├─ conftest.py
│  ├─ schema_in.json
│  └─ test_config.py
├─ webapp
│  ├─ static
│  │  ├─ css
│  │  │  └─ main.css
│  │  └─ sctipt
│  │     └─ index.js
│  └─ templates
│     ├─ 404.html
│     ├─ base.html
│     └─ index.html
└─ wine_quality_prediction_app_screenshot.png
```

## 6. Project 6 - End-to-end-category-prediction

This repository contains an end-to-end project for predicting categories. The dataset is preprocessed & transformed. The category selection is done. An experiment is done using the BERT , XLM-Roberta Model is used to predict the categories.

### Project Structure

Project Organization
------------
```bash

category-prediction               # Root directory of the project
├── .dvc                          # DVC directory for versioning data and models
├── .gitignore                    # Git ignore file to specify ignored files and directories
├── config                        # Configuration file
├── .dvcignore                    # DVC ignore file for specifying files to ignore in DVC
├── .github                       # GitHub directory
│   └── workflows                 # GitHub workflows directory
│       └── ci-cd.yaml            # Continuous integration and deployment configuration
├── Dockerfile                    # Dockerfile for building Docker image
├── LICENSE                       # License file
├── Makefile                      # Makefile for defining and running tasks
├── README.md                     # Readme file with project description
├── app.py                        # Python script for the main application
├── dag.py                        # Python script for defining a Directed Acyclic Graph (DAG)
├── data_given                    # Directory for given data
│   ├── .gitignore                # Git ignore file for data_given directory
│   ├── en_au.csv.dvc             # DVC file for versioning en_au.csv
│   ├── en_nz.csv.dvc             # DVC file for versioning en_nz.csv
│   ├── ...                       # Other data files versioned using DVC
├── docker-compose.yml            # Docker Compose file for defining multi-container Docker applications
├── docs                          # Documentation directory
│   ├── Makefile                  # Makefile for building documentation
│   ├── commands.rst              # Commands documentation file
│   ├── conf.py                   # Configuration file for Sphinx documentation generator
│   ├── getting-started.rst       # Getting started documentation file
│   ├── index.rst                 # Index file for documentation
│   └── make.bat                  # Windows batch file for building documentation
├── ml_logs.log                   # Log file for ML logging
├── models                        # Directory for storing trained models
│   └── .gitkeep                  # Git keep file to keep models directory in Git
├── notebooks                     # Directory for Jupyter notebooks
│   ├── .gitkeep                  # Git keep file to keep notebooks directory in Git
│   ├── 1.0-initial-data-exploration.ipynb  # Jupyter notebook for initial data exploration
│   ├── 2.0-initial-data-exploration.ipynb  # Another Jupyter notebook for initial data exploration
│   └── 3.0-llm-category-extraction.ipynb  # Jupyter notebook for category extraction
├── params.yaml                   # Parameters configuration file in YAML format
├── prediction_service            # Directory for prediction service code
│   ├── __init__.py               # Python package initialization file
│   ├── model                     # Directory for storing trained model file
│   │   └── model.joblib          # Trained model file
│   ├── prediction.py             # Python script for prediction service
│   └── schema_in.json            # JSON schema for input data
├── references                    # Directory for reference files
│   └── .gitkeep                  # Git keep file to keep references directory in Git
├── reports                       # Directory for reports
│   ├── .gitkeep                  # Git keep file to keep reports directory in Git
│   └── figures                    # Directory for report figures
│       ├── .gitkeep              # Git keep file to keep figures directory in Git
│       ├── category_top_50_values.png        # Report figure
│       ├── category_wordcloud.png            # Report figure
│       ├── correlation_matrix_heatmap.png    # Report figure
│       ├── locale_top_50_values.png          # Report figure
│       ├── product_brand_top_50_values.png   # Report figure
│       └── product_name_top_50_values.png    # Report figure
├── params.json                   # Parameters JSON file
├── scores.json                   # Scores JSON file
├── unique_category_values.txt    # Text file for unique category values
├── requirements.txt              # Requirements file for specifying project dependencies
├── saved_models                  # Directory for saved models
│   └── model.joblib              # Saved model file
├── setup.py                      # Setup script for packaging the project
├── src                           # Source code directory
│   ├── __init__.py               # Python package initialization file
│   ├── __pycache__               # Python bytecode cache directory
│   │   ├── get_data.cpython-38.pyc           # Bytecode cache file
│   │   └── train_and_evaluate.cpython-38.pyc  # Bytecode cache file
│   ├── analyse.py                # Python script for data analysis
│   ├── analyse_data.py           #
│   ├── catgeory_llm.py           # Python script for category LLM (Local Linear Model)
│   ├── get_data.py               # Python script for getting data
│   ├── load_data.py              # Python script for loading data
│   ├── mlflow_logging.py         # Python script for MLflow logging
│   ├── split_data.py             # Python script for splitting data
│   ├── train_and_evaluate.py     # Python script for training and evaluating models
│   └── transform_data.py         # Python script for transforming data
├── test_environment.py           # Python script for testing environment setup
├── tox.ini                       # Configuration file for tox testing tool
└── webapp                        # Directory for web application
    ├── .gitkeep                  # Git keep file to keep webapp directory in Git
    ├── static                    # Static files directory for web application
    │   ├── css                   # CSS files directory
    │   │   └── main.css          # Main CSS file
    │   └── sctipt                # Script files directory
    │       └── index.js          # JavaScript file for index page
    └── templates                 # HTML templates directory for web application
        ├── 404.html              # HTML template for 404 error page
        ├── base.html             # Base HTML template
        └── index.html            # HTML template for index page

```

Feel free to explore the repositories for more detailed information on each project. If you have any questions or need further information, please contact me at [jojoyadav255@gmail.com].
