# read the data from data source
# save it in the data/raw for further process
import os
import sys
import argparse 
import warnings
import json
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('src'))
from wordcloud import WordCloud
from get_data import read_params, print_tabulated_data,setup_logging
warnings.filterwarnings("ignore")

def explore_dataframe(df):
    # Explore the DataFrame and print various statistics.
    logging.info("******************************Shape of the  DataFrame...******************************")
    print(df.shape)
    logging.info("******************************Description of the  DataFrame..******************************")
    print(df.describe())
    logging.info("******************************Print all the columns of the DataFrame...******************************")
    print(df.columns)
    logging.info("******************************Information of the DataFrame...******************************")
    print(df.info())
    logging.info("******************************Data Types of the DataFrame...******************************")
    print(df.dtypes)
    logging.info("******************************Missing values of the DataFrame..******************************")
    print(df.isnull().sum())
    logging.info("******************************NaN values of the DataFrame...******************************")
    print(df.isna().sum())
    logging.info("******************************Null values of the DataFrame...******************************")
    print(df.isna().sum())
    logging.info("******************************Dublicate of the DataFrame...******************************")
    print(df[df.duplicated()])


def word_cloud(df, column_name, folder_path):
    # Generate and save a word cloud from a specified column in the DataFrame.
    text = " ".join(df[column_name].astype(str))
    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        min_font_size=1
    ).generate(text)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f"{column_name}_wordcloud.png")

    # Display and save the generated word cloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(file_path)
    plt.close()


def frequency_encoding_and_correlation(df,folder_path, plot=False):
    # Perform frequency encoding on specified textual columns, compute the correlation matrix, and save the plot in a folder.
    textual_columns = ['product_name', 'locale', 'category', 'product_brand']
    for col in textual_columns:
        df[col + '_encoded'] = df[col].map(df[col].value_counts())
    correlation_matrix = df[[col + '_encoded' for col in textual_columns]].corr()
    if plot:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Textual Columns')
        plot_path = os.path.join(folder_path, 'correlation_matrix_heatmap.png')
        plt.savefig(plot_path)
        plt.close()

def unique_values(df):
    #Print count of unique values for each column in the DataFrame.
    for column_name in df.columns:
        unique_count = df[column_name].nunique()
        print(f"Number of unique values in {column_name}: {unique_count}")


def unique_percentage(df):
    # Calculate and print the percentage of unique values for each column in the DataFrame.
    for column_name in df.columns:
        unique_pct = (df[column_name].nunique() / df[column_name].count()) * 100
        print(f"Percentage of Unique values in {column_name}: {unique_pct:.2f}%")

def plot_count(df, save_folder):
    #Plot top 50 values from each column and save the plots as PNG files.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for column in df.columns:
        if df[column].dtype == 'object':
            top_values = df[column].value_counts().nlargest(50)
            plt.figure(figsize=(12, 6))
            top_values.plot(kind='bar')
            plt.title(f'Top 50 Values for {column}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.tight_layout()
            # Save the plot as PNG file
            save_path = os.path.join(save_folder, f'{column}_top_50_values.png')
            plt.savefig(save_path)
            plt.close()

def analyse_data(config_path):
    print("Start Analysing data...")
    config = read_params(config_path)
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    plots_dir = config["reports"]["plots"]
    files = config["reports"]["files"]
    df = pd.read_csv(raw_data_path,sep=",")
    logging.info("Explore DataFrame...")
    explore_dataframe(df)
    logging.info("Explore Unique Values DataFrame...")
    unique_values(df)
    logging.info("Explore Top 50 Values DataFrame...")
    plot_count(df,plots_dir)
    logging.info("Explore Unique Percentage Values for each column of the Dataframe...")
    unique_percentage(df)
    logging.info("HeatMap of the Dataframe...")
    frequency_encoding_and_correlation(df,plots_dir, plot=True)
    logging.info("WordCloud of the Dataframe...")
    word_cloud(df, 'category', plots_dir)
    logging.info("All the plots available in Reports/figures...")
    logging.info("Finish Analysing data...")


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    logger = setup_logging()
    analyse_data(config_path=parsed_args.config)  