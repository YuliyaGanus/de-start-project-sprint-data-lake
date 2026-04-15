from datetime import datetime
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
import os

# Указываем базовый путь к твоим скриптам
BASE_PATH = '/lessons/scripts'

default_args = {
    'owner': 's26586465',
    'start_date': datetime(2022, 1, 1), # Дата начала архива данных
    'retries': 1 # Попытка перезапуска при сбое
}

dag = DAG(
    dag_id='s26586465_project_final',
    default_args=default_args,
    schedule_interval='@daily', # Запуск раз в сутки
    catchup=False # Не запускать за все прошлые даты сразу
)

# 1. Задача для витрины пользователей
user_mart = SparkSubmitOperator(
    task_id='user_mart',
    application=f'{BASE_PATH}/user_mart.py',
    conn_id='spark_default',
    application_args=[],
    conf={
        "spark.driver.maxResultSize": "2g",
        "spark.execution.memory": "2g"
    },
    dag=dag
)

# 2. Задача для витрин гео-зон и активности
geo_zones = SparkSubmitOperator(
    task_id='geo_zones',
    application=f'{BASE_PATH}/geo_zones.py',
    conn_id='spark_default',
    dag=dag
)

# 3. Задача для рекомендаций друзей
friend_rec = SparkSubmitOperator(
    task_id='friend_recommendations',
    application=f'{BASE_PATH}/friend_recommendation.py',
    conn_id='spark_default',
    dag=dag
)

# Настраиваем последовательность выполнения
user_mart >> geo_zones >> friend_rec