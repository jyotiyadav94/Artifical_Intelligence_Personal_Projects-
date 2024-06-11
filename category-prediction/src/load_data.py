import os
import sys
import argparse
import warnings
import logging
sys.path.append(os.path.abspath('src'))
from get_data import read_params, get_data, print_tabulated_data,setup_logging
warnings.filterwarnings("ignore")

def load_and_save(config_path):
    # Loads data, prints a tabulated view, and saves it to a specified path.
    logger.info("Start loading and Saving data...")
    config = read_params(config_path)
    df = get_data(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    logger.info("Finish loading...")
    print_tabulated_data(df)
    df.to_csv(raw_data_path, sep=",", index=False)
    logger.info("Saving data to data/raw folder...")


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    logger = setup_logging()
    load_and_save(config_path=parsed_args.config)
