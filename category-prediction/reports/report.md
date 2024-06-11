#Project Report

## Project Overview

The aim of this project is to develop a machine learning model capable of predicting product categories based on product names and brands. Two advanced models, BERT and XLM-RoBERTa, have been fine-tuned.

## End-to-End Repository
The project constitutes an end-to-end repository, encompassing the following key steps:

1. **Data Preparation**: Involves the gathering and preprocessing of data necessary for model training.
2. **Data Analysis**: Entails a thorough examination of the dataset to gain insights and understand its characteristics.
3. **Data Transformations**: Covers the conversion and manipulation of data to prepare it for training.
4. **Training and Evaluation**: Encompasses the actual training of the model and evaluating their performance.
5. **MLFLOW Logging**: Incorporates MLFLOW for tracking experiments and managing model artifacts.
6. **DVC (Data Version Control)**: Utilizes DVC for versioning data and model, ensuring reproducibility and scalability.
7. **Prediction Service**: Includes the development of a web application for deploying the trained model for real-time predictions.
8. **Docker & docker-compose.yaml**: Utilizes Docker and docker-compose.yaml for containerization and orchestration of the application environment.
9. **Integration with CI/CD Pipeline using GitHub Actions**: Integrates the project with a continuous integration and continuous deployment (CI/CD) pipeline using GitHub Actions for automated testing, building, and deployment processes.
10. **Directed Acyclic Graph (DAG) Definition**: A `dag.py` file has been created to define the execution flow of scripts. The DAG specifies the sequence of execution.


## Project Organization
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

## Data Preparation

`get_data` Class: At first we loads and combines data from multiple CSV files into a single DataFrame. It reads parameters from a YAML configuration file, retrieves the data source directory path, lists CSV files in the directory, reads each CSV file into a DataFrame, concatenates them, and prints a tabulated view of the combined DataFrame.

##### Methods
- `read_params(config_path)`: Reads parameters from a YAML config file.
- `get_data(config_path)`: Loads and combines data from CSV files into a DataFrame.
- `print_tabulated_data(data_frame)`: Prints a tabulated view of a DataFrame.

`load_data` Class:
This class loads data using the `get_data` class & saves the data to a specified path.

#### Methods:
- `load_and_save(config_path)`: Loads data, prints a tabulated view, and saves it to a specified path.


## Data Analysis

#### Summary Statistics:

- **Shape**: The dataset comprises 112,293 rows and 6 columns.

- **Column Descriptions**:

    - **product_name**: Names of the products.Each entry represents a specific product.

    - **brand_id**: Unique identifiers for the brands associated with the products wrt region. Indicates the brand to which each product belongs. 

    - **locale**: Geographical or regional locale for each product. Indicates the languages, location/region where the product is relevant or available.

    - **category_id**: Unique identifiers for the categories to which the products belong wrt region. Each category is assigned a numerical ID.

    - **category**: Denotes the category to which each product belongs. Provides information about the type or classification of the product.

    - **product_brand**: Names of the brands associated with the products. Provides information about the brand of each product listed in the dataset. 

- **Brand and Category IDs**:
  - `brand_id` has 49,700 non-null values.
  - `category_id` has 112,282 non-null values.

- **Missing Values**:
  - `brand_id`: 62,593 missing values.
  - `category_id`: 11 missing values.
  - `category` and `product_brand`: 11 and 26,912 missing values respectively.

#### Data Types:
- The DataFrame consists of 2 float64 columns (`brand_id` and `category_id`) and 4 object columns (`product_name`, `locale`, `category`, and `product_brand`).

#### Unique Values:
- **Columns**:
  - `product_name`: 90,171 unique values (80.30% of total).
  - `brand_id`: 4,434 unique values (8.92% of total).
  - `locale`: 13 unique values (0.01% of total).
  - `category_id`: 7,981 unique values (7.11% of total).
  - `category`: 10,365 unique values (9.23% of total).
  - `product_brand`: 19,309 unique values (22.62% of total).

1. **Data Distribution**:
   - The dataset contains a diverse range of product categories and brands across different locales.
   - Most products have unique names (`product_name`).
   - There is a considerable number of missing values in `brand_id` and `product_brand` columns.

2. **Data Quality**:
   - The presence of missing values in `brand_id`, `category_id`, `category`, and `product_brand` columns suggests potential data quality issues.
   - The percentage of unique values in `brand_id` and `product_brand` indicates a high level of diversity in brands.

3. **Potential Areas for Further Analysis**:
   - Investigate the distribution and patterns of missing values.
   - Explore the relationship between product categories and brands.
   - Identify and handle duplicate data, if any.
   - Perform additional preprocessing steps to handle missing values and ensure data quality.


![category_top_50_values](https://github.com/jyotiyadav94/category-prediction/assets/72126242/b8313e2f-ab7d-464e-869f-9393985be66e)

The bar plot illustrates the top 50 values within a category, sorted by frequency (count). It visualizes the distribution of values and their occurrence frequency.

**Insights**:
* Skewed Distribution: The plot shows a right-skewed distribution, indicating that a few values have significantly higher counts than others. This suggests dominance of a few popular items in the category.
* Top Values: "Vino" (wine) emerges as the most frequent value, followed by "juegos" (games) and "cerveza/birra" (beer).

![locale_top_50_values](https://github.com/jyotiyadav94/category-prediction/assets/72126242/746a2ffb-f047-4f9d-b9f2-d8eea2037fa8)

The bar plot shows the top 50 values for the "locale" variable, representing language and regional settings. It indicates the count or frequency of each locale, with the x-axis displaying different locale values.

**Insights**:
* Dominant Locales: "it_it" (Italian in Italy) and "es_es" (Spanish in Spain) are the most common, suggesting a significant presence of users from these regions.
* Distribution: The plot is heavily skewed to the right, with a long tail of less frequent locales after the dominant ones.
* Language Variety: Multiple Spanish and Portuguese locales imply a diverse user base from various countries.
* Other Locales: Presence of locales like "fr_fr" (French in France) and "en_au" (English in Australia) indicates international usage.


![product_brand_top_50_values](https://github.com/jyotiyadav94/category-prediction/assets/72126242/46c2d649-cee6-4673-93fa-1829355e3788)

The bar plot displays the top 50 product brands by their frequency within a dataset, likely representing sales data or consumer reviews.

**Insights**:

* Distribution: The plot is heavily right-skewed, indicating a few dominant brands and many less frequent ones, typical in consumer goods markets.
* Top Brands: Carrefour leads by a large margin, followed by Conad and Auchan, major European supermarket chains.
* Brand Variety: A wide range of brands across categories like food (Nestle, Danone), personal care (Colgate, Nivea), and electronics (Samsung, LG) is observed.


![product_name_top_50_values](https://github.com/jyotiyadav94/category-prediction/assets/72126242/1783c9dd-f9d2-46af-bd75-f9825fee266d)

The bar plot shows the top 50 product names by their frequency count. It indicates the most commonly occurring product names, likely from retail sales or inventory records.

**Insights**:
* Distribution: The plot is heavily right-skewed, with a few top products dominating the count, typical in consumer goods data.
* Top Products: "Cerveza" (beer) leads, followed by "birra," "whisky," "vodka," and "shampoo," suggesting diverse product categories.
* Language: Primarily in Italian and Portuguese, implying origins from Italy or Brazil.

![category_wordcloud](https://github.com/jyotiyadav94/category-prediction/assets/72126242/f99e0dd2-6a47-4265-a05b-641486db7c96)

The word cloud visually represents text data, with each word's size indicating its frequency or importance. It focuses on consumer goods and shopping behavior, particularly food and beverages.
**Insights**:
* Prominent Words: Key terms like "vino" (wine), "cerveza" (beer), "carne

![correlation_matrix_heatmap](https://github.com/jyotiyadav94/category-prediction/assets/72126242/4918423b-a3ac-4d85-972d-2000dc2b3ff0)


## Data Preprocessing & Data Transformation

### Data Preprocessing:

- **Deletes unnecessary columns:** Removes `brand_id` and `category_id`.
- **Removes rows with missing values:** Eliminates rows with missing values in `product_name`, `category`, or `product_brand`.
- **Standardizes column values:** Converts text-based column values to lowercase and removes leading/trailing whitespaces.

### Category Filtering:

- **Filters categories:** Retains categories with a count greater than a specified threshold (default: 50).

### Conflicting Data Removal:

- **Removes conflicting data:** Groups data by `product_brand` and `product_name`, eliminating rows with inconsistent category assignments within each group.

### Duplicate Removal:

- **Removes duplicate rows:** Identifies and removes duplicate rows to ensure data integrity.

### Unique Category Extraction:

- **Extracts unique categories:** Saves them to a JSON file for reference.

### Category Translation:

- **Translates category names:** Converts them to English based on locale information.
- **Removes language and locale columns:** After translation, removes redundant columns.

### Final Category Replacement:

- **Replaces categories:** Utilizes mappings from a JSON file to replace category names.

### Final Data Saving:

- **Saves transformed DataFrame:** Writes the processed DataFrame to a CSV file for further analysis.

We have two dataframes from the above 
1. Dataframe with category (300 categories translated into English)
2. Dataframe with category (category defined by LLM)

- At the end we end with both final dataframes with the size of (35297,3).

## Category Reduction using LLM 

This script employs OpenAI's GPT-3 turbo model to categorize items.

API Key Configuration: It sets up the API key required to access the OpenAI GPT-3 model

Item Categorization:
- For each distinct item in the dataset, the script utilizes the GPT-3 model to categorize it.
- The model is presented with the name of each item and a predefined list of categories to choose from.

Data Storage:
- Categorized items and their respective categories are organized into a dictionary.
- This dictionary is then saved into a JSON file, serving as a reference for future use.

## Training and Evaluation

This script is designed to train and evaluate a BERT model for categorizing items. Here's a breakdown of its key functionalities:

**Data Processing:**

- The script reads the data from a CSV file, extracting text and label information.
- It tokenizes the text data using a BERT tokenizer.

**Model Architecture:**

- The BERT-based classifier model is defined, with a dropout layer and a linear layer for classification.

**Training Loop:**
- It trains the BERT model using the provided dataset.
- Training is conducted over multiple epochs, with each epoch iterating through the dataset.
- During training, the model's weights are updated based on the calculated loss using backpropagation.

**Evaluation:**

- After each epoch, the model's performance is evaluated on a validation dataset.
- Metrics such as precision, recall, F1-score, and accuracy are computed to assess the model's performance.
- Since our classes are imbalanced and we prioritize correctly identifying positive cases (recall) or minimizing false positives (precision), we should consider F1-metric for the evaluation of model. 

## MLflow logging

Here we have focused on integrating MLflow for experiment tracking and model management.

MLflow Setup:

- MLflow is initialized with the specified remote server URI and experiment name. This enables tracking and organizing experiments on the MLflow server.

MLflow Run:
- We have MLflow run with the configured run name and logs all relevant parameters using `mlflow.log_param()`.
- This ensures that all the parameters used during training are captured and associated with the current MLflow run.

Logging Training Metrics:

- During each epoch of training, metrics such as precision, recall, F1-score, and accuracy are computed.
- These metrics are then logged to MLflow using `mlflow.log_metric()` to track the model's performance over the course of training.

Model Saving with MLflow:

- After training completes, the trained model is saved as an MLflow artifact.
- If MLflow is connected to a remote tracking server, the model is logged with the specified registered model name using `mlflow.pytorch.log_model()`.
- If MLflow is using the local filesystem for tracking, the model is saved locally as a PyTorch model file using `mlflow.pytorch.save_model()`.

MLflow Run Completion:

- Once the training and model saving process is complete, the MLflow run is marked as completed.
- All logged metrics and parameters are associated with the current MLflow run, making it easy to track and compare experiments later.

<img width="1439" alt="Screenshot 2024-06-10 at 17 27 39" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/ad0eb705-3ea9-49e6-8d96-b26db9f8edc5">
<img width="1439" alt="Screenshot 2024-06-10 at 17 24 01" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/16418b05-fadc-49bb-b4b2-6c7ae12da016">
<img width="1439" alt="Screenshot 2024-06-10 at 17 23 35" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/b82a46ab-5690-49c0-9586-90414f28c195">
<img width="1439" alt="Screenshot 2024-06-10 at 17 23 15" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/2fb5415c-dc14-49c0-ab8e-44fdf57bf72b">


## Model Results 

## BERT Model Validation Results with 25 different labels

The model is trained on the below categories. Which have been analysed and generated using Closed source LLM OpenAI API as defined in the class catgeory_llm.py. 

```bash
id2label={0: 'Household', 1: 'Dairy', 2: 'galletas', 3: 'Beverages', 4: 'Meat/Poultry/Seafood', 5: 'bombones', 6: 'Other', 7: 'AlcoholicBeverages', 8: 'Snacks/Candy', 9: 'Canned/JarredGoods', 10: 'PersonalCare', 11: 'FrozenFoods', 12: 'Pasta/Grains', 13: 'Bakery', 14: 'Prepared/Ready-Made Foods', 15: 'Toys: Other', 16: 'Prepared/Ready-Made_Foods', 17: 'Electronics', 18: 'Baby', 19: 'Pet', 20: 'FreshProduce', 21: 'Toys', 22: 'Produce', 23: 'Vitamins: Other', 24: 'Medicines: Household'}
```

### Parameter 
bert_model_name = 'bert-base-uncased'
num_classes = length
max_length = 128
batch_size = 16
num_epochs = 4
learning_rate = 2e-5
test_size=0.2
random_state=42

## Model Validation Results
The Model was trained on GPU see the notebooks for reference of the results below. "4.0-BERT-model-training.ipynb"

**Validation Accuracy:** 0.8822

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.93      | 0.95   | 0.94     | 772     |
| 1     | 0.91      | 0.94   | 0.92     | 649     |
| 2     | 0.98      | 0.86   | 0.91     | 69      |
| 3     | 0.92      | 0.92   | 0.92     | 843     |
| 4     | 0.89      | 0.93   | 0.91     | 620     |
| 5     | 0.95      | 0.83   | 0.89     | 24      |
| 6     | 0.77      | 0.70   | 0.73     | 177     |
| 7     | 0.95      | 0.94   | 0.94     | 945     |
| 8     | 0.79      | 0.85   | 0.82     | 717     |
| 9     | 0.89      | 0.88   | 0.88     | 397     |
| 10    | 0.91      | 0.87   | 0.89     | 519     |
| 11    | 0.58      | 0.56   | 0.57     | 105     |
| 12    | 0.82      | 0.89   | 0.86     | 225     |
| 13    | 0.79      | 0.81   | 0.80     | 333     |
| 14    | 0.45      | 0.28   | 0.34     | 18      |
| 15    | 0.00      | 0.00   | 0.00     | 14      |
| 16    | 0.76      | 0.66   | 0.70     | 151     |
| 17    | 0.86      | 0.94   | 0.90     | 32      |
| 18    | 0.92      | 0.98   | 0.95     | 50      |
| 19    | 0.96      | 0.96   | 0.96     | 224     |
| 20    | 0.91      | 0.91   | 0.91     | 54      |
| 21    | 0.00      | 0.00   | 0.00     | 30      |
| 22    | 0.90      | 1.00   | 0.95     | 18      |
| 23    | 0.65      | 0.88   | 0.75     | 42      |
| 24    | 0.53      | 0.25   | 0.34     | 32      |

**Accuracy:** 0.88  
**Macro Avg Precision:** 0.76  
**Macro Avg Recall:** 0.75  
**Macro Avg F1:** 0.75  
**Weighted Avg Precision:** 0.88  
**Weighted Avg Recall:** 0.88  
**Weighted Avg F1:** 0.88


## XLM-Roberta Model Validation Results with 25 different labels
**Validation Accuracy:** 0.8636

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.93      | 0.95   | 0.94     | 772     |
| 1     | 0.87      | 0.91   | 0.89     | 649     |
| 2     | 1.00      | 0.84   | 0.91     | 69      |
| 3     | 0.92      | 0.90   | 0.91     | 843     |
| 4     | 0.87      | 0.92   | 0.89     | 620     |
| 5     | 1.00      | 0.83   | 0.91     | 24      |
| 6     | 0.77      | 0.69   | 0.73     | 177     |
| 7     | 0.96      | 0.93   | 0.94     | 945     |
| 8     | 0.71      | 0.82   | 0.76     | 717     |
| 9     | 0.88      | 0.86   | 0.87     | 397     |
| 10    | 0.89      | 0.88   | 0.89     | 519     |
| 11    | 0.56      | 0.42   | 0.48     | 105     |
| 12    | 0.82      | 0.84   | 0.83     | 225     |
| 13    | 0.70      | 0.78   | 0.74     | 333     |
| 14    | 0.00      | 0.00   | 0.00     | 18      |
| 15    | 0.00      | 0.00   | 0.00     | 14      |
| 16    | 0.83      | 0.64   | 0.72     | 151     |
| 17    | 0.80      | 0.88   | 0.84     | 32      |
| 18    | 0.98      | 0.92   | 0.95     | 50      |
| 19    | 0.95      | 0.93   | 0.94     | 224     |
| 20    | 0.89      | 0.78   | 0.83     | 54      |
| 21    | 0.00      | 0.00   | 0.00     | 30      |
| 22    | 0.95      | 1.00   | 0.97     | 18      |
| 23    | 0.67      | 0.88   | 0.76     | 42      |
| 24    | 0.69      | 0.28   | 0.40     | 32      |

**Accuracy:** 0.86  
**Macro Avg Precision:** 0.75  
**Macro Avg Recall:** 0.72  
**Macro Avg F1:** 0.72  
**Weighted Avg Precision:** 0.86  
**Weighted Avg Recall:** 0.86  
**Weighted Avg F1:** 0.86


## BERT Model with 320 labels

```bash
id2label = {0: 'suavizante', 1: 'leche', 2: 'lavadoras', 3: 'galletas', 4: 'coffee', 5: 'jamón', 6: 'bombones', 7: 'queso', 8: 'drinks', 9: 'salchichas', 10: 'aceite de girasol', 11: 'cerveza', 12: 'chocolate', 13: 'aceite', 14: 'Margarine', 15: 'mozzarella', 16: 'yogurt', 17: 'broth', 18: 'mayonesa', 19: 'shampoo', 20: 'papel higiénico', 21: 'helados', 22: 'detergent', 23: 'cereales', 24: 'rice', 25: 'toallitas', 26: 'juegos', 27: 'deodorant', 28: 'compresas', 29: 'pan', 30: 'whisky', 31: 'smart tv', 32: 'couches', 33: 'eau', 34: 'poulet', 35: 'chocolats', 36: 'plats cuisinés', 37: 'desserts', 38: 'jouets', 39: 'vin', 40: 'jeux', 41: 'chips', 42: 'snacks', 43: 'Coke', 44: 'jambon', 45: 'livres', 46: 'boissons', 47: 'champagne', 48: 'fromage', 49: 'beurre', 50: 'saucisses', 51: 'alimentation', 52: 'lessive en capsules', 53: 'jus', 54: 'viande', 55: 'bonbons', 56: 'bière blonde', 57: 'saumon fumé', 58: 'yaourt', 59: 'biscuits', 60: 'foie gras de canard', 61: 'dentifrice', 62: 'Pizza', 63: 'vodka', 64: 'lessive liquide', 65: 'gin', 66: 'glace', 67: 'rhum', 68: 'sauces', 69: 'chocolat', 70: 'surgelés', 71: 'bière', 72: 'plats préparés', 73: 'nettoyants', 74: 'crème', 75: 'gel douche', 76: 'emmental', 77: 'gel', 78: 'pâtes', 79: 'chorizo', 80: 'shampoing', 81: 'smartphones', 82: 'chocolat au lait', 83: 'gorgonzola', 84: 'ketchup', 85: 'teddy', 86: 'mascarpone', 87: 'gambas', 88: 'lasagne', 89: 'bacon', 90: 'prosecco', 91: 'panettone', 92: 'cacao', 93: 'samsung galaxy', 94: 'notebook', 95: 'vino tinto', 96: 'turrón', 97: 'aceitunas', 98: 'vino', 99: 'patatas fritas', 100: 'agua', 101: 'pañales', 102: 'earphones', 103: 'limpiadores', 104: 'liquor', 105: 'vino blanco', 106: 'atún', 107: 'hamburguesas', 108: 'pan de molde', 109: 'salsas', 110: 'detergente lavavajillas', 111: 'cat food', 112: 'yogur', 113: 'comida para perros', 114: 'paté para perros', 115: 'Juices', 116: 'postres', 117: 'ginebra', 118: 'dentífrico', 119: 'ron', 120: 'carne', 121: 'ensaladas', 122: 'pechuga de pavo', 123: 'conservas', 124: 'juguetes', 125: 'mermelada', 126: 'accesorios cocina', 127: 'champú', 128: 'frigorífico combi', 129: 'pasta', 130: 'platos preparados', 131: 'liquid detergent', 132: 'energy drink', 133: 'atún claro', 134: 'coffee capsules', 135: 'tablet', 136: 'tomate frito', 137: 'smartwatch', 138: 'aceite de oliva', 139: 'led tv', 140: 'aceite de oliva virgen extra', 141: 'dry fruits', 142: 'pollo', 143: 'fresh paste', 144: 'nectar', 145: 'frozen', 146: 'papel de cocina', 147: 'café molido', 148: 'perfumes', 149: 'passata di pomodoro', 150: 'merendine', 151: 'salmone affumicato', 152: 'cibo per gatti', 153: 'biscotti', 154: 'mayonnaise', 155: 'cibi pronti', 156: 'cioccolatini', 157: 'dash detersivo', 158: 'the', 159: 'crema viso', 160: 'tonno rio mare', 161: "pasta all'uovo", 162: 'frutta secca', 163: 'marmellata', 164: 'panna', 165: 'cibo per cani', 166: 'detersivo lavastoviglie', 167: 'piselli', 168: 'latte parzialmente scremato', 169: 'sgrassatore', 170: 'mortadella', 171: 'pesto', 172: 'birra', 173: 'tonno', 174: 'piadine', 175: 'petto di pollo', 176: 'biscotti mulino bianco', 177: 'bevande analcoliche', 178: 'acqua', 179: 'vino bianco', 180: 'salviettine', 181: 'tinte capelli', 182: 'assorbenti lines', 183: 'dentifricio', 184: 'formaggio', 185: 'bibite', 186: 'torrone', 187: 'bresaola', 188: 'farmacia', 189: 'giochi per bambini', 190: 'aceto balsamico', 191: 'ammorbidente', 192: 'succhi di frutta', 193: 'wurstel', 194: 'salsa', 195: 'detergente intimo', 196: 'caffè', 197: 'cura dei capelli', 198: 'formaggio spalmabile', 199: 'pancetta', 200: 'spumante', 201: 'frollini', 202: 'patatine fritte', 203: 'capsule caffè', 204: 'zuppe', 205: 'salami', 206: 'croissant', 207: 'bagnoschiuma', 208: 'filetti di merluzzo', 209: 'parmigiano', 210: 'cesti natalizi', 211: 'detersivi', 212: 'detersivo lavatrice', 213: 'detersivo piatti', 214: 'crema al cioccolato', 215: 'vino rosso', 216: 'rotoli di carta', 217: 'pasta di semola', 218: 'burro', 219: 'snack', 220: 'cioccolato', 221: 'caramelle', 222: 'conserve', 223: 'pasta sfoglia', 224: 'dessert', 225: 'olio di semi', 226: 'pasticceria', 227: 'brodo', 228: 'alici', 229: 'bagno doccia', 230: 'prosciutto cotto', 231: 'smartphone', 232: 'surgelati', 233: 'olive', 234: 'tisane', 235: 'assorbenti', 236: 'liquore', 237: 'olio extravergine di oliva', 238: 'hamburger', 239: 'insalata', 240: 'pasta ripiena', 241: 'carta igienica', 242: 'pomodori pelati', 243: 'gamberi', 244: 'uova', 245: 'alimenti', 246: 'pecorino', 247: 'pulizie di casa', 248: 'pane', 249: 'prosciutto crudo', 250: 'profumatori ambiente', 251: 'grissini', 252: 'pulizia pavimenti', 253: 'riso', 254: 'patatine', 255: 'deodorante', 256: 'farina', 257: 'pulizia del viso', 258: 'stracchino', 259: 'sapone liquido', 260: 'sughi per pasta', 261: 'speck', 262: 'pulizia bagno', 263: 'ricotta', 264: 'amari', 265: 'accessori cucina', 266: 'crackers', 267: 'fette biscottate', 268: 'cereali', 269: 'minestrone', 270: 'cereali kelloggs', 271: 'latte', 272: 'salsicce', 273: 'tovaglioli', 274: 'rasoio', 275: 'salumi', 276: 'rum', 277: 'spinaci', 278: 'lego', 279: 'medicine', 280: 'body care', 281: 'spirits', 282: 'skin care', 283: 'pharmacy', 284: 'beer', 285: 'vitamins', 286: 'supplements', 287: 'fridge', 288: 'laptops', 289: 'tyres', 290: 'wine', 291: 'medicines', 292: 'absorbent', 293: 'biscuit and biscuit', 294: 'paper towel', 295: 'chocolate milk', 296: 'Toothpaste', 297: 'wines', 298: 'juice', 299: 'softener', 300: 'toilet paper', 301: 'Powder detergent', 302: 'chocolates', 303: 'pharmaceutical products', 304: 'sunscreen', 305: 'water', 306: 'noodle', 307: 'sausage', 308: 'insecticide', 309: 'diapers', 310: 'disinfectant', 311: 'Coconut Water', 312: 'dog food', 313: 'milk', 314: 'Tomato Sauce', 315: 'cookies', 316: 'soap', 317: 'bread', 318: 'cheeses', 319: 'games'}
```
```bash
Validation Accuracy: 0.6953
              precision    recall  f1-score   support

           0       1.00      0.88      0.94        25
           1       0.80      0.85      0.82        41
           2       0.95      0.95      0.95        21
           3       0.94      0.86      0.89        69
           4       0.57      0.96      0.71        45
           5       0.48      0.82      0.61        39
           6       0.95      0.83      0.89        24
           7       0.94      0.89      0.91        99
           8       0.42      0.62      0.50        82
           9       0.89      0.89      0.89        19
          10       0.86      0.50      0.63        12
          11       0.96      0.97      0.96        96
          12       0.49      0.95      0.65        56
          13       0.00      0.00      0.00        14
          14       0.92      0.92      0.92        12
          15       0.85      0.93      0.89        30
          16       0.84      0.79      0.82        53
          17       1.00      0.73      0.84        11
          18       0.89      0.73      0.80        11
          19       0.91      0.91      0.91        54
          20       0.72      0.93      0.81        46
          21       1.00      0.42      0.59        12
          22       0.35      0.26      0.30        35
          23       0.73      0.73      0.73        30
          24       0.97      0.94      0.95        31
          25       1.00      0.75      0.86        12
          26       0.29      0.71      0.41        65
          27       0.82      0.93      0.87        30
          28       0.95      0.90      0.93        21
          29       0.67      0.80      0.73        20
          30       0.94      0.91      0.92        64
          31       0.78      1.00      0.88        53
          32       1.00      0.62      0.77         8
          33       0.93      0.81      0.87        16
          34       0.71      1.00      0.83        12
          35       0.03      0.07      0.05        28
          36       0.00      0.00      0.00         5
          37       0.64      0.33      0.44        27
          38       0.00      0.00      0.00        14
          39       0.91      0.96      0.93       288
          40       0.32      0.85      0.46        61
          41       0.74      0.85      0.79        33
          42       0.24      0.36      0.29        45
          43       0.59      0.83      0.69        12
          44       0.91      0.84      0.87        37
          45       0.00      0.00      0.00        16
          46       0.00      0.00      0.00        22
          47       0.89      0.83      0.86        29
          48       0.57      0.85      0.68        84
          49       1.00      0.93      0.96        14
          50       1.00      0.69      0.82        13
          51       0.00      0.00      0.00        11
          52       0.83      0.83      0.83        12
          53       1.00      0.50      0.67        12
          54       0.00      0.00      0.00         7
          55       1.00      0.58      0.74        12
          56       0.00      0.00      0.00        14
          57       1.00      1.00      1.00        11
          58       0.72      0.78      0.75        27
          59       0.70      0.62      0.65        26
          60       0.91      0.95      0.93        22
          61       1.00      0.80      0.89         5
          62       0.85      0.95      0.90        41
          63       0.96      1.00      0.98        24
          64       0.94      0.94      0.94        16
          65       0.89      1.00      0.94        17
          66       0.64      0.54      0.58        13
          67       0.90      0.60      0.72        15
          68       0.67      0.50      0.57        12
          69       0.27      0.52      0.35        33
          70       0.00      0.00      0.00        16
          71       0.67      0.89      0.76        38
          72       0.00      0.00      0.00        26
          73       0.00      0.00      0.00         7
          74       0.70      0.70      0.70        10
          75       0.90      0.69      0.78        13
          76       0.75      1.00      0.86         6
          77       0.74      0.85      0.79        20
          78       0.75      0.38      0.50         8
          79       1.00      0.96      0.98        23
          80       1.00      0.81      0.90        16
          81       0.45      0.76      0.57        17
          82       0.00      0.00      0.00        11
          83       1.00      0.95      0.97        19
          84       0.91      0.91      0.91        23
          85       0.67      0.46      0.55        13
          86       1.00      0.83      0.91         6
          87       0.92      0.80      0.86        15
          88       0.75      1.00      0.86         9
          89       0.60      0.50      0.55         6
          90       0.92      0.86      0.89        28
          91       0.88      0.97      0.92        30
          92       0.90      0.60      0.72        15
          93       0.65      0.73      0.69        15
          94       0.60      0.60      0.60        10
          95       0.70      0.88      0.78        57
          96       1.00      0.93      0.97        15
          97       1.00      0.87      0.93        15
          98       0.68      0.82      0.75       235
          99       0.73      0.67      0.70        12
         100       0.60      1.00      0.75        18
         101       0.81      0.96      0.88        27
         102       0.90      0.82      0.86        11
         103       0.85      0.88      0.87        26
         104       1.00      0.60      0.75        10
         105       0.73      0.81      0.77        27
         106       0.46      0.81      0.59        16
         107       1.00      0.79      0.88        19
         108       1.00      0.90      0.95        10
         109       0.69      0.77      0.73        26
         110       0.88      0.93      0.90        30
         111       0.71      0.89      0.79        46
         112       0.46      0.75      0.57        24
         113       0.65      0.73      0.69        33
         114       0.00      0.00      0.00        14
         115       0.74      0.72      0.73        32
         116       1.00      0.33      0.50        18
         117       0.92      0.80      0.86        15
         118       0.00      0.00      0.00        11
         119       0.75      0.75      0.75         8
         120       0.36      0.48      0.41        52
         121       0.89      0.73      0.80        11
         122       1.00      0.91      0.95        11
         123       0.00      0.00      0.00        15
         124       0.00      0.00      0.00        30
         125       1.00      0.94      0.97        16
         126       0.00      0.00      0.00        10
         127       1.00      0.73      0.84        33
         128       1.00      0.83      0.91        12
         129       0.37      0.67      0.48        42
         130       0.00      0.00      0.00        13
         131       0.51      0.76      0.61        37
         132       0.94      0.67      0.78        24
         133       0.00      0.00      0.00         9
         134       0.00      0.00      0.00        15
         135       1.00      0.67      0.80         6
         136       0.79      0.79      0.79        14
         137       1.00      0.80      0.89        10
         138       0.37      1.00      0.54        14
         139       0.00      0.00      0.00        11
         140       0.00      0.00      0.00        10
         141       1.00      0.38      0.56        13
         142       0.37      0.72      0.49        18
         143       0.00      0.00      0.00         7
         144       0.83      0.77      0.80        13
         145       0.00      0.00      0.00        25
         146       0.82      0.90      0.86        10
         147       0.00      0.00      0.00        12
         148       0.89      0.86      0.88        29
         149       0.54      0.86      0.67        22
         150       0.00      0.00      0.00        11
         151       0.95      0.95      0.95        19
         152       0.73      0.87      0.80        54
         153       0.35      0.78      0.48        41
         154       0.93      0.87      0.90        15
         155       0.00      0.00      0.00        28
         156       0.00      0.00      0.00        32
         157       0.50      0.12      0.20         8
         158       0.86      0.95      0.90        19
         159       0.23      0.64      0.34        11
         160       0.00      0.00      0.00         9
         161       0.60      0.69      0.64        13
         162       0.00      0.00      0.00        15
         163       0.86      0.60      0.71        20
         164       0.94      0.89      0.91        18
         165       0.77      0.85      0.81        65
         166       0.55      0.86      0.67        21
         167       1.00      1.00      1.00         9
         168       0.50      1.00      0.67        11
         169       1.00      0.50      0.67        10
         170       1.00      1.00      1.00        18
         171       0.94      0.94      0.94        16
         172       0.81      0.71      0.76        35
         173       0.81      0.93      0.87        28
         174       1.00      0.79      0.88        14
         175       1.00      0.93      0.96        14
         176       0.00      0.00      0.00         6
         177       0.79      0.71      0.75        48
         178       1.00      1.00      1.00        27
         179       0.00      0.00      0.00        17
         180       0.94      0.89      0.92        19
         181       1.00      0.64      0.78        14
         182       0.80      0.92      0.86        13
         183       0.65      0.94      0.77        18
         184       0.54      0.82      0.65        83
         185       0.00      0.00      0.00         9
         186       0.88      1.00      0.93         7
         187       1.00      1.00      1.00         8
         188       0.50      0.05      0.09        21
         189       0.42      0.41      0.41        32
         190       1.00      0.83      0.91        12
         191       0.95      1.00      0.97        19
         192       0.62      0.70      0.66        30
         193       0.93      1.00      0.96        13
         194       1.00      0.13      0.24        15
         195       1.00      0.89      0.94         9
         196       0.86      0.79      0.82        38
         197       0.00      0.00      0.00         7
         198       1.00      0.45      0.62        11
         199       0.79      0.83      0.81        18
         200       0.88      0.75      0.81        40
         201       0.50      0.60      0.55         5
         202       0.00      0.00      0.00        15
         203       0.84      0.90      0.87        30
         204       1.00      0.38      0.55        16
         205       0.60      0.78      0.67        36
         206       0.93      0.93      0.93        14
         207       1.00      0.38      0.56        13
         208       1.00      0.83      0.91        12
         209       0.71      0.83      0.77        12
         210       0.00      0.00      0.00        15
         211       0.00      0.00      0.00        19
         212       0.65      0.88      0.74        48
         213       0.81      0.93      0.87        14
         214       0.00      0.00      0.00         7
         215       0.00      0.00      0.00        19
         216       0.81      0.89      0.85        19
         217       0.56      0.56      0.56         9
         218       0.83      1.00      0.91        15
         219       0.29      0.51      0.37        37
         220       0.46      0.77      0.57        43
         221       0.90      0.64      0.75        14
         222       0.38      0.35      0.36        26
         223       0.00      0.00      0.00         6
         224       1.00      0.12      0.22        32
         225       0.00      0.00      0.00         7
         226       0.50      0.04      0.08        23
         227       0.00      0.00      0.00        10
         228       0.88      0.88      0.88         8
         229       0.62      0.71      0.67         7
         230       0.62      0.91      0.74        35
         231       0.00      0.00      0.00        12
         232       0.14      0.22      0.17        27
         233       0.90      1.00      0.95        18
         234       0.68      0.94      0.79        16
         235       1.00      0.38      0.55         8
         236       0.80      0.29      0.42        14
         237       0.66      1.00      0.79        23
         238       0.72      0.87      0.79        15
         239       0.91      0.77      0.83        26
         240       0.00      0.00      0.00        20
         241       0.95      0.95      0.95        19
         242       0.00      0.00      0.00        14
         243       0.83      0.71      0.77         7
         244       1.00      1.00      1.00        19
         245       0.00      0.00      0.00        17
         246       0.95      1.00      0.97        19
         247       0.43      0.65      0.52        23
         248       0.81      0.71      0.76        24
         249       1.00      0.54      0.70        26
         250       0.00      0.00      0.00        19
         251       1.00      0.31      0.47        13
         252       0.93      0.56      0.70        25
         253       0.82      0.93      0.87        15
         254       0.36      0.90      0.51        10
         255       0.75      1.00      0.86        15
         256       0.93      0.93      0.93        15
         257       0.00      0.00      0.00         6
         258       1.00      1.00      1.00         4
         259       0.92      1.00      0.96        12
         260       0.89      0.85      0.87        20
         261       0.90      0.90      0.90        10
         262       1.00      0.14      0.25         7
         263       0.76      1.00      0.87        13
         264       1.00      0.60      0.75         5
         265       0.00      0.00      0.00        15
         266       1.00      0.93      0.96        14
         267       0.64      0.88      0.74         8
         268       0.60      0.60      0.60        10
         269       0.79      0.92      0.85        12
         270       0.00      0.00      0.00         7
         271       0.00      0.00      0.00         9
         272       1.00      0.92      0.96        12
         273       1.00      1.00      1.00         7
         274       1.00      0.83      0.91        12
         275       0.00      0.00      0.00        13
         276       0.88      0.54      0.67        13
         277       1.00      0.75      0.86         8
         278       0.83      0.91      0.87        11
         279       0.00      0.00      0.00        16
         280       0.00      0.00      0.00        12
         281       0.60      0.67      0.63        18
         282       1.00      0.06      0.11        17
         283       0.42      0.63      0.51        62
         284       0.84      1.00      0.91        47
         285       0.53      0.95      0.68        42
         286       0.00      0.00      0.00         8
         287       1.00      0.94      0.97        18
         288       0.50      0.57      0.53         7
         289       0.73      0.80      0.76        10
         290       0.90      0.56      0.69        16
         291       0.43      0.31      0.36        32
         292       1.00      1.00      1.00        10
         293       0.25      0.78      0.38         9
         294       0.91      1.00      0.95        10
         295       1.00      0.92      0.96        13
         296       1.00      0.83      0.91        12
         297       1.00      0.89      0.94        35
         298       0.91      1.00      0.95        10
         299       1.00      1.00      1.00        18
         300       0.00      0.00      0.00        17
         301       1.00      0.77      0.87        13
         302       0.00      0.00      0.00        27
         303       0.00      0.00      0.00         9
         304       1.00      0.75      0.86         8
         305       0.00      0.00      0.00         9
         306       0.85      0.94      0.89        18
         307       1.00      0.91      0.95        11
         308       1.00      1.00      1.00         2
         309       0.96      0.96      0.96        23
         310       1.00      0.80      0.89         5
         311       1.00      1.00      1.00         2
         312       0.69      0.75      0.72        12
         313       1.00      0.83      0.91        12
         314       0.76      0.93      0.84        14
         315       0.60      0.12      0.20        25
         316       1.00      0.91      0.95        22
         317       1.00      0.71      0.83         7
         318       1.00      1.00      1.00        18
         319       0.00      0.00      0.00        18

    accuracy                           0.70      7060
   macro avg       0.65      0.63      0.62      7060
weighted avg       0.67      0.70      0.66      7060
```


## Prediction Service

<img width="1439" alt="Screenshot 2024-06-10 at 10 54 10" src="https://github.com/jyotiyadav94/category-prediction/assets/72126242/5e3a6a69-934e-41ae-9e23-c7aca193f99c">

**Note:**
- Due to my old laptop's hardware limitations and GPU access issues,
    - I had to train the model on colab and then integrate it to the report mentioned above.
    - While working on pipelines, I used my local machine to train the model on a very small portion of data.
- For demo purposes, I have just attached the relevant code without any model. If models are needed please feel free to reach out I would gladly address the issue.
- I had also deployed the model on AWS but due to billing issues, I removed the EC2 instances from AWS. (screenshot attached).

