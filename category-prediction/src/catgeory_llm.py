import os
import sys
import json
import openai
import logging
import argparse
import pandas as pd
from get_data import read_params
sys.path.append(os.path.abspath('src'))
from dotenv import load_dotenv, find_dotenv
from get_data import print_tabulated_data,setup_logging

# Configure logging
logging.basicConfig(filename='ml_logs.log', filemode='a', level=logging.INFO)
logger = logging.getLogger(__name__)
config = find_dotenv(".env")
load_dotenv()
MY_OPEN_API_KEY = os.getenv("MY_OPEN_API_KEY")

# These can be also handled using Open LLM 
# After careful observation we came to a conclusion that the final categories which can be further processed for the BERT model
categories = [
        "FreshProduce",
        "Meat/Poultry/Seafood",
        "Dairy",
        "FrozenFoods",
        "Bakery",
        "Snacks/Candy",
        "Prepared/Ready-Made_Foods",
        "Beverages",
        "AlcoholicBeverages",
        "Pasta/Grains",
        "Canned/JarredGoods",
        "Household",
        "PersonalCare",
        "Baby",
        "Pet",
        "Other"
        ]

def query_response(model_, prompt):
    # Queries the OpenAI GPT-3 model and returns the response.
    completion = openai.ChatCompletion.create(
        model=model_,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return completion.choices[0].message.content

def categorize_item(item_name, model):
    # Categorizes a single item using the GPT-3 model.
    prompt = f'''
    Reply only what you are asked, nothing else. Provide a category of a given item by choosing only one value. 
    Categorize this item: '{item_name}'. Choose one category from the list: {', '.join(categories)}
    '''
    return query_response(model, prompt)

def categorize_items(transform_data_path, model,column):
    print_tabulated_data(transform_data_path)
    # Categorizes items in the CSV file using the GPT-3 model.
    unique_categories = transform_data_path[column].unique().tolist()

    # Convert unique categories to dictionary keys with default value None
    val = {key: None for key in unique_categories}
    # Iterate over the keys using index to save API costs
    for key in val.keys():
        if val[key] is None:
            val[key] = categorize_item(key, model)
    return val

def save_categories(val, final_category):
    # Saves the categorized items to a JSON file.
    with open(final_category, 'w') as file:
        json.dump(val, file, indent=4)
    logging.info("Categorized items saved to JSON file.")

def categorize_data(config_path):
    # Categorizes data using the OpenAI GPT-3 model.
    config = read_params(config_path)
    final_category = config["reports"]["final_category"]
    transform_data_path = config["transformed_data"]["final_dataframe"]
    model = config["open_ai_model"]
    df = pd.read_csv(transform_data_path,sep=",")
    # Load API key from environment variables
    openai.api_key = MY_OPEN_API_KEY
    if not openai.api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    logger.info("Data categorization started....")
    val = categorize_items(df, model,'category')
    logger.info("Saving the Data categorization...")
    save_categories(val, final_category)
    logger.info("Data categorization completed.")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    logger = setup_logging()
    categorize_data(config_path=parsed_args.config)
