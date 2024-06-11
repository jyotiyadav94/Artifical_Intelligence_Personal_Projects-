import os
import sys
import yaml
import logging
import argparse
import warnings
import pandas as pd
from tabulate import tabulate
sys.path.append(os.path.abspath('src'))
warnings.filterwarnings("ignore")

# Configure logging
def setup_logging(log_file='ml_logs.log'):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Set logging level for handlers
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def read_params(config_path):
    # Read parameters from the specified YAML config file.
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def print_tabulated_data(data_frame):
    # Print tabulated data frame.
    logger.info("\n" + tabulate(data_frame.head(), headers='keys', tablefmt='psql'))


def get_data(config_path):
    # Load and combine data from all the CSV files.
    logger.info("Start Getting the data...")
    config = read_params(config_path)
    data_path = config["data_source"]["source"]
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    combined_df = pd.DataFrame()
    # Read and append the contents of each CSV file to the list
    for file in csv_files:
        file_path = os.path.join(data_path, file)
        logger.info(f'Reading {file}')
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], axis=0)
    logger.info("Combining all the dataframe..")
    
    # Comment this to train the model on whole dataset
    combined_df = combined_df.head(20)
    print_tabulated_data(combined_df)
    logger.info("Finish getting the data...")
    return combined_df

logger = setup_logging()
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
