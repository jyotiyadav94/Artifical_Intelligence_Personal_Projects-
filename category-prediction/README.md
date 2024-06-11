# End-to-end-category-prediction

This repository contains an end-to-end project for predicting categories. The dataset is preprocessed & transformed. The category selection is done. An experiment is done using the BERT , XLM-Roberta Model is used to predict the categories.For more detailed information see the [Report](https://github.com/jyotiyadav94/category-prediction/blob/main/reports/report.md).

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

### Run locally
Clone the project:

```bash
 git clone https://github.com/jyotiyadav94/category-prediction.git
```

Create and activate an environment: 

```bash
conda create -n your_env_name python=3.8 -y
```

```bash
conda activate your_env_name
```

Install the requirements for this project:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset we use can be found.

Download it and put it into the data_given/ directory.

### Initialize Git and DVC and track data

```bash
git init
```
```bash
dvc init 
```
```bash
dvc add data_given/*.csv
```

## Demo 
The Demo can be Found Below

```bash
https://huggingface.co/spaces/Jyotiyadav/CategoryPrediction
```

<img width="1437" alt="Screenshot 2024-06-10 at 23 52 16" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/67f93012-5659-4925-b1d3-36f0b0f4e711">

<img width="1437" alt="Screenshot 2024-06-10 at 23 52 21" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/f91bb0a4-e11a-4cd6-b04a-fb60e12e2a4e">


### Run the DVC Pipeline

The pipeline includes the following stages:
1) Combining data
2) loading data
3) analyse data
4) category selection (using Open api)
4) transform data
5) train and evaluate data
6) mlflow logging

Run the DVC Pipeline:
```bash
dvc repro
```
### Run Tests
These tests ensure that the code handles different scenarios correctly:
- Producing valid results for correct input ranges.
Run the tests:

```bash
pytest
```
### Automatic testing

Each time you perform a push or pull request, automatic testing of your code is triggered using the ci_cd.yaml in the .github/workflows directory.


## Run the category Prediction application

<img width="1439" alt="Screenshot 2024-06-10 at 10 54 10" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/5e3a6a69-934e-41ae-9e23-c7aca193f99c">


### Running the app and the inference model


```bash
python app.py 
```

### Set up and run the app with Docker

The Docker container is configured to handle POST requests.

1) open Docker Desktop application 

2) Build the Docker container:
```bash
docker-compose build
```

3) Run the Docker container:
```bash
docker-compose up -d
```

4) Open your preferred browser and navigate to http://127.0.0.1:8080/ to start using the application.


## Run the MLflow Training Model

To start MLFLOW run the server using the following command
```bash
mlflow server --host 127.0.0.1 --port 1234
```

In another terminal, run the following command
```bash
python src/log_production_model.py
```
Navigate to the MLflow UI at http://localhost:1234/
<img width="1439" alt="Screenshot 2024-06-10 at 17 27 39" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/6dce7107-5897-4d10-ae51-639996b28b42">


## AWS-CICD-Deployment-with-Github-Actions
The model is deployed using EC2 instance by following the below steps. 

<img width="1437" alt="Screenshot 2024-06-10 at 20 29 11" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/d51ac2c9-7afc-4e47-b6b9-508b10d216ca">
<img width="1437" alt="Screenshot 2024-06-10 at 20 28 46" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/d757f69c-40f0-4b07-86a3-194a741401f0">


#### 1. Login to AWS console.
#### 2. Create IAM user for deployment

```bash
#with specific access
1. EC2 access : It is virtual machine
2. ECR: Elastic Container registry to save your docker image in aws

#Description: About the deployment

1. Build docker image of the source code
2. Push your docker image to ECR
3. Launch Your EC2 
4. Pull Your image from ECR in EC2
5. Lauch your docker image in EC2

#Policy:
1. AmazonEC2ContainerRegistryFullAccess
2. AmazonEC2FullAccess
```

#### 3. Create ECR repo to store/save docker image
```bash
- Save the URI: 136566696263.dkr.ecr.us-east-1.amazonaws.com/mlproject
```
#### 4. Create EC2 machine (Ubuntu)
#### 5. Open EC2 and Install docker in EC2 Machine:

```bash
#optinal
sudo apt-get update -y
sudo apt-get upgrade
#required
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```
#### 6. Configure EC2 as self-hosted runner:

### 7. Setup github secrets:
```bash
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION = us-east-1
AWS_ECR_LOGIN_URI = 
ECR_REPOSITORY_NAME = 
```
