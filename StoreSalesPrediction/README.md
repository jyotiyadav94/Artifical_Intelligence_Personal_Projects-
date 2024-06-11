# Project: Store Sales Prediction in Ecuador

## Project Introduction
This project revolves around predicting sales for thousands of product families available at Favorita stores in Ecuador. The provided dataset comprises various features such as dates, store and product information, promotional status, and sales figures. The goal is to accurately forecast sales based on these attributes.

## Main Goal
The primary objective of this project is to enhance the Kaggle score for store sales prediction in Ecuador through the following comparison:
* Employing conventional machine learning models such as Linear Regression, Random Forest, and XGBoost.
* Fine-tuning an open-source Large Language Model (LLM) like FLANT-5.
 
## Kaggle Submission Results
![Alt text](<Images/Screenshot 2024-03-22 at 21.13.38.png>)

## HuggingFace Demo
[Link to Kaggle Inference](https://huggingface.co/spaces/Jyotiyadav/Forecasting_model)

## Model 
[Link to Saved Model](https://huggingface.co/Jyotiyadav/model2.0)

### File Descriptions and Data Field Information
Find more information about the dataset [here](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).
- `train.csv`: Contains the training data, consisting of time series information on `store_nbr`, `family`, `onpromotion`, and the target `sales`.
  - `store_nbr`: Identifies the store where products are sold.
  - `family`: Identifies the type of product sold.
  - `sales`: Represents the total sales for a product family at a specific store on a given date. Fractional values are possible.
  - `onpromotion`: Indicates the total number of items in a product family being promoted at a store on a given date.
- `test.csv`: Contains the test data with the same features as the training data. Predictions for target sales are required for the dates in this file.
- `sample_submission.csv`: A sample submission file provided in the correct format.
- `stores.csv`: Metadata about stores including city, state, type, and cluster.
  - `cluster`: Groups similar stores together.
- `oil.csv`: Daily oil price data, including values during both the train and test data timeframes. Given that Ecuador is oil-dependent, fluctuations in oil prices can significantly impact its economic health.
- `holidays_events.csv`: Holidays and Events data with metadata. Special attention needs to be paid to the `transferred` column which indicates holidays officially moved to another date by the government. Additional holidays are days added to regular calendar holidays.

This comprehensive dataset provides an opportunity to explore various factors influencing sales and to develop predictive models that can provide valuable insights for Favorita stores in Ecuador.

## Project Structure for Conventional Machine Learning 
1. **Imports**: Import necessary libraries and modules.
2. **Reading the Dataset**: Read each of the files containing the dataset.
3. **Preprocessing**: Perform data preprocessing tasks such as handling missing values, encoding categorical variables, etc.
4. **Exploratory Data Analysis (EDA)**: Utilize libraries like Sweetviz, Autoviz, and pandas profiling for comprehensive EDA to gain insights into the data.
5. **Feature Engineering**: Create new features or modify existing features to improve model performance.
6. **Saving the features in MongoDB Atlas**: Store the dataset in MongoDB Atlas for further use.
7. **Train Test Split**: Split the dataset into training and testing sets for model training and evaluation.
8. **Scaling & One-Hot Encoding Features**: Scale numerical features and perform one-hot encoding on categorical features as required for model compatibility.
9. **Modelling**: Build predictive models using machine learning algorithms.
10. **Saved the Model**: Save the trained model for future use and reproducibility.
11. **Submission**: Generate predictions using the trained model and prepare the submission file.


## LLM Model FLAN-T5

FLAN-T5 is an open-source, sequence-to-sequence language model developed by Google researchers in late 2022. It is capable of performing various natural language processing tasks and can be used both in research and commercial applications. The model is based on the Transformer architecture and trained on a large corpus of text known as the Colossal Clean Crawled Corpus (C4).

### Fine-Tuning
Fine-tuning FLAN-T5 is essential to adapt it to specific tasks and improve its performance. This process allows customization of the model according to the user's needs and data, making it accessible to a wider range of users, including smaller organizations and individual researchers without GPU resources.
Instruction fine-tuning is a technique used to customize pre-trained large language models (LLMs) to perform specific tasks based on explicit instructions or demonstrations. Unlike traditional fine-tuning, which involves training a model on task-specific data, instruction fine-tuning provides higher-level guidance to the model to shape its behavior according to predefined instructions.

**Instruction:** Analyze the sentiment of the text and determine whether it's positive or negative.
**Input Text:** "I thoroughly enjoyed the movie; it was fantastic."
**Output Text:** Positive, Negative

### Potential Applications
Potential applications of fine-tuned FLAN-T5 include:
- **Text Summarization:** Condensing large amounts of text into concise summaries.
- **Text Classification:** Categorizing text data into predefined classes (e.g., spam or non-spam, positive or negative sentiment).


## Project Structure for fine-tuning LLM Model FLAN-T5
[LLM-Fine tuned]([https://huggingface.co/Jyotiyadav/model2.0](https://github.com/jyotiyadav94/StoreSalesPrediction/blob/main/LLM_fine_tuning/LLM_fine_tuningipynb.ipynb)https://github.com/jyotiyadav94/StoreSalesPrediction/blob/main/LLM_fine_tuning/LLM_fine_tuningipynb.ipynb).
