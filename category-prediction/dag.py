import os 
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
sys.path.append(os.path.abspath('category-prediction'))
from src.load_data import load_and_save
from src.get_data import read_params,print_tabulated_data,get_data
from src.analyse_data import explore_dataframe,word_cloud,save_unique_category
from src.analyse_data import frequency_encoding_and_correlation,unique_values
from src.train_and_evaluate import unique_percentage,plot_count,analyse_data
from src.train_and_evaluate import CategoryClassificationDataset,BERTClassifier
from src.transform_data import delete_columns,translate_words,translate_dataframe
from src.train_and_evaluate import get_device,train_and_evaluate,id2label_mapping
from src.train_and_evaluate import save_model,evaluate,load_data,train
import subprocess

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 20),
    'email': ['jojoyadav79811@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def run_script(script_name):
    """
    Runs a Python script using subprocess.
    """
    result = subprocess.run(["python", script_name], check=True, capture_output=True, text=True)
    print(f"Output of {script_name}:\n{result.stdout}")
    return result.stdout

dag = DAG(
    'sequential_scripts_dag',
    default_args=default_args,
    description='A simple DAG to run scripts sequentially',
    schedule_interval=timedelta(days=1),
)

run_script1 = PythonOperator(
    task_id='get_data',
    python_callable=get_data,
    op_args=['get_data.py'],
    dag=dag,
)

run_script2 = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_args=['load_data'],
    dag=dag,
)

run_script3 = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    op_args=['transform_data'],
    dag=dag,
)


run_script4 = PythonOperator(
    task_id='train_and_evaluate',
    python_callable=train_and_evaluate,
    op_args=['train_and_evaluate'],
    dag=dag,
)

run_script1 >> run_script2 >> run_script3>> run_script4
