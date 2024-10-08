from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import pickle
import json
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from typing import Any, Dict, Literal
import logging
import io
from datetime import datetime, timedelta
import boto3
from botocore.client import Config

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Настройка логирования
log = logging.getLogger()
log.addHandler(logging.StreamHandler())

# Получение переменных из Airflow для Yandex S3
s3_endpoint_url = Variable.get("yandex_s3_endpoint_url")
s3_access_key = Variable.get("yandex_s3_access_key")
s3_secret_key = Variable.get("yandex_s3_secret_key")
s3_bucket = Variable.get("yandex_s3_bucket")

features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

models = {
    "logistic_regression": LogisticRegression(),
    "gradient_boosting": GradientBoostingClassifier(),
    "decision_tree": DecisionTreeClassifier()
}

# Функция для создания клиента Yandex S3
def create_s3_client():
    return boto3.client(
        's3',
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        config=Config(signature_version='s3v4')
    )

def create_dag(dag_id: str, model_name: Literal["gradient_boosting", "logistic_regression", "decision_tree"]):

    def init(model_name: str) -> Dict[str, Any]:
        dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.info(f'Pipeline started: {dt}')
        metrics = {
            'model_name': model_name,
            'start_dt': dt
        }
        return metrics

    def get_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='init')

        metrics['download_dt_begin'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dataset = fetch_openml(name='adult', version=2)
        data = pd.concat([dataset["data"], pd.DataFrame(dataset["target"], columns=['target'])], axis=1)

        metrics['download_dt_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['dataset_shape'] = list(data.shape)

        s3_client = create_s3_client()
        filebuffer = io.BytesIO()
        data.to_pickle(filebuffer)
        filebuffer.seek(0)
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=f"arsenii_korol/{metrics['model_name']}/datasets/adult.pkl",
            Body=filebuffer
        )
        log.info('downloaded data.')
        return metrics

    def process_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='get_data')

        s3_client = create_s3_client()
        response = s3_client.get_object(
            Bucket=s3_bucket, 
            Key=f"arsenii_korol/{metrics['model_name']}/datasets/adult.pkl"
        )
        
        df = pd.read_pickle(io.BytesIO(response['Body'].read()))
        X, y = df[features], df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
        
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)

        for name, data in zip(
            ["X_train", "X_test", "y_train", "y_test"],
            [X_train_processed, X_test_processed, y_train, y_test]
        ):
            filebuffer = io.BytesIO()
            pickle.dump(data, filebuffer)
            filebuffer.seek(0)
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=f"arsenii_korol/{metrics['model_name']}/datasets/{name}.pkl",
                Body=filebuffer
            )

        metrics['processed_dt_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['features'] = features
        log.info('processed data.')
        return metrics

    def train_model(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids='process_data')

        s3_client = create_s3_client()
        data = {}
        for data_name in ["X_train", "X_test", "y_train", "y_test"]:
            response = s3_client.get_object(
                Bucket=s3_bucket, 
                Key=f"arsenii_korol/{metrics['model_name']}/datasets/{data_name}.pkl"
            )
            data[data_name] = pd.read_pickle(io.BytesIO(response['Body'].read()))
        
        metrics['train_dt_begin'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model = models[metrics['model_name']]
        model.fit(data["X_train"], data["y_train"])

        probas = model.predict_proba(data["X_test"])[:, 1]
        y_predicted = (probas > 0.5).astype(int)
        y_true = data["y_test"]
        
        model_eval = {
            'precision': precision_score(y_true, y_predicted),
            'f1_score': f1_score(y_true, y_predicted),
            'recall': recall_score(y_true, y_predicted)
        }
        
        metrics['metrics'] = model_eval
        metrics['train_dt_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.info('trained model.')
        return metrics

    def save_results(**kwargs) -> None:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids='train_model')

        s3_client = create_s3_client()
        buff = io.BytesIO()
        buff.write(json.dumps(metrics, indent=2).encode())
        buff.seek(0)
        s3_client.put_object(
            Bucket=s3_bucket, 
            Key=f"arsenii_korol/{metrics['model_name']}/metrics/metrics.json", 
            Body=buff
        )

    default_args = {
        "owner": "arsenii_korol",
        'email': 'arsenykingrus@gmail.com',
        "retry": 2,
        "retry_delay": timedelta(minutes=1)
    }

    # Создание DAG
    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        default_args=default_args
    )

    with dag:
        task_init = PythonOperator(task_id="init", python_callable=init, dag=dag, op_kwargs={"model_name": model_name})
        task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag)
        task_process_data = PythonOperator(task_id="process_data", python_callable=process_data, dag=dag)
        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
        task_save_results = PythonOperator(task_id="save_result", python_callable=save_results, dag=dag)

        task_init >> task_get_data >> task_process_data >> task_train_model >> task_save_results

for model_name in models.keys():
    create_dag(f"arsenii_korol_{model_name}", model_name)
