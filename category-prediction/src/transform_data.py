import os
import sys
import json
import argparse
import logging 
import warnings
import pandas as pd
sys.path.append(os.path.abspath('src'))
from deep_translator import GoogleTranslator
sys.path.append(os.path.abspath('category-prediction'))
from get_data import read_params, print_tabulated_data,setup_logging
from analyse_data import explore_dataframe
warnings.filterwarnings("ignore")

def delete_columns(data_frame, columns_to_delete):
    # Drop the specified columns from the DataFrame
    modified_df = data_frame.drop(columns=columns_to_delete, errors='ignore')
    return modified_df

def translate_category(df, category_column, locale_column, folder_path):
    def translate_words(word_list, source_language, target_language="en"):
        # Translate the words from the source language to the target language
        word_dict = {}
        for word in set(word_list):
            word_dict[word] = GoogleTranslator(source=source_language, target=target_language).translate(word)
        return word_dict

    # Extract the language (first two characters of the locale)
    df['language'] = df[locale_column].str[:2]

    # Extract unique category and language pairs
    unique_pairs = df[[category_column, 'language']].drop_duplicates()

    # Translate categories to English
    translations = {}
    for lang in unique_pairs['language'].unique():
        if lang != 'en':
            subset = unique_pairs[unique_pairs['language'] == lang]
            translated_categories = translate_words(subset[category_column], source_language=lang)
            translations.update(translated_categories)
        else:
            for category in unique_pairs[category_column]:
                translations[category] = category

    category_translation_dict = {category: translations[category] for category in unique_pairs[category_column]}
    # replace translations with eng versions
    print("categories, translation: ", category_translation_dict)
    df[category_column] = df[category_column].map(lambda x: category_translation_dict.get(x, x))
    file_path = os.path.join(folder_path, "translated.json")
    with open(file_path, 'w') as json_file:
        json.dump(category_translation_dict, json_file, indent=4)
    print_tabulated_data(df)
    return df

def filter_categories(df, target_column, count_threshold=50):
    # Filters the DataFrame to keep only rows where the count of the target column values is greater than the specified threshold.
    value_counts = df[target_column].value_counts()
    values_to_keep = value_counts[value_counts > count_threshold].index
    filtered_df = df[df[target_column].isin(values_to_keep)]
    return filtered_df

def save_unique_categories(df, column_name, json_file_path):
    # Extracts unique categories from the specified column in a DataFrame and saves them to a JSON file.
    unique_categories = df[column_name].unique().tolist()
    # Save to JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(unique_categories, json_file, indent=4)


def replace_categories(df, json_file_path):
    # Replace categories in the DataFrame using mappings from the JSON file.
    with open(json_file_path, 'r') as json_file:
        mappings = json.load(json_file)
    # Replace categories using mappings
    df['category'].replace(mappings, inplace=True)

    return df


def replace_category_with_english(df, folder_path):
    # Replace the words in the 'category' column of the DataFrame with their corresponding English words.
    # Load mappings from JSON file
    file_path = os.path.join(folder_path, "translated.json")
    with open(file_path, 'r') as file:
        mappings = json.load(file)
    # Replace category with English words based on mappings
    df['category'].replace(mappings, inplace=True)
    return df

def transform_data(config_path):
    logger.info("Start transforming data...")
    config = read_params(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    translated_data_path = config["transformed_data"]["translated_dataframe"]
    final_category = config["reports"]["final_category"]
    unique_category = config["reports"]["files"]
    translated_report = config["reports"]["translated_data"]
    final_data_path = config["transformed_data"]["final_dataframe"]
    filter_count = config["transformed_data"]["filter_count"]
    df = pd.read_csv(raw_data_path,sep=",")

    logger.info("Delete columns...")
    df = delete_columns(df, ['brand_id','category_id'])

    logger.info("Remove nan's...")
    df.dropna(subset=['product_name', 'category', 'product_brand'], inplace=True)

    logger.info("Converting column values to lowercase...")
    df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

    # saving categories whose count >50
    filtered_df = filter_categories(df, 'category', filter_count)
    explore_dataframe(filtered_df)

    logger.info("Removing conflicting data (Different Categories with unique Brands and Products)...")
    filtered_df = filtered_df.groupby(['product_brand', 'product_name']).filter(lambda x: len(x['category'].unique()) == 1)

    logger.info("Remove duplicate...")
    filtered_df.drop_duplicates(inplace=True)

    save_unique_categories(filtered_df, 'category', unique_category)
    
    logger.info("translating category and deleting helper column 'locale'...")
    filtered_df = translate_category(filtered_df, 'category', 'locale', translated_report)
    filtered_df = delete_columns(filtered_df, ['locale','language'])
    filtered_df = replace_category_with_english(filtered_df, translated_report)
    filtered_df.to_csv(translated_data_path, sep=",", index=False)

    logger.info("translating category and deleting helper column 'locale'...")
    filtered_df = replace_categories(filtered_df, final_category)

    logger.info("Saving final dataframe..")
    filtered_df.to_csv(final_data_path, sep=",", index=False)
    print("Finish transforming data...")


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    logger = setup_logging()
    transform_data(config_path=parsed_args.config)  