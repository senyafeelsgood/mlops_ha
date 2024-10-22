import mlflow
import os
import pandas as pd
import boto3
from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score, recall_score
from typing import Any, Dict, Literal
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pickle
import io
import logging
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger()
log.addHandler(logging.StreamHandler())

BUCKET = Variable.get("S3_BUCKET")
AWS_ENDPOINT_URL = Variable.get("AWS_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")

s3_client = boto3.client(
    's3',
    endpoint_url=AWS_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

models = {
    "logistic_regression": LogisticRegression(),
    "gradient_boosting": GradientBoostingClassifier(),
    "decision_tree": DecisionTreeClassifier()
}

features = ['age', 'fnlwgt', 'education-num',
            'capital-gain', 'capital-loss', 'hours-per-week']

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

def create_dag(dag_id: str, model_name: Literal["gradient_boosting", "logistic_regression", "decision_tree"]):

    def init(model_name: str) -> Dict[str, Any]:
        dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.info(f'Pipeline started: {dt}')
        metrics = {
            'model_name': model_name,
            'start_dt': dt
        }
        configure_mlflow()
        
        experiment_name = f"Experiment_{model_name}"
        experiment_id = mlflow.create_experiment(experiment_name)
        metrics['experiment_id'] = experiment_id
        
        return metrics

    def get_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='init')

        dataset = fetch_openml(name='adult', version=2, as_frame=True)
        X, y = dataset['data'], dataset['target']
        y = y.map({'<=50K': 0, '>50K': 1})

        data = pd.concat([X, y.rename('target')], axis=1)
        metrics['dataset_shape'] = list(data.shape)

        filebuffer = io.BytesIO()
        data.to_pickle(filebuffer)
        filebuffer.seek(0)
        s3_client.upload_fileobj(
            Fileobj=filebuffer,
            Bucket=BUCKET,
            Key=f"datasets/{metrics['model_name']}/adult.pkl"
        )

        return metrics

    def process_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='get_data')
        filebuffer = io.BytesIO()
        s3_client.download_fileobj(BUCKET, f"datasets/{metrics['model_name']}/adult.pkl", filebuffer)
        filebuffer.seek(0)
        df = pd.read_pickle(filebuffer)
        
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
            s3_client.upload_fileobj(
                Fileobj=filebuffer,
                Bucket=BUCKET,
                Key=f"datasets/{metrics['model_name']}/{name}.pkl"
            )

        return metrics

    def train_model(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='process_data')

        data = {}
        for data_name in ["X_train", "X_test", "y_train", "y_test"]:
            filebuffer = io.BytesIO()
            s3_client.download_fileobj(BUCKET, f"datasets/{metrics['model_name']}/{data_name}.pkl", filebuffer)
            filebuffer.seek(0)
            data[data_name] = pickle.load(filebuffer)

        model = models[metrics['model_name']]
        model.fit(data["X_train"], data["y_train"])
        
        with mlflow.start_run(experiment_id=metrics['experiment_id'], run_name=metrics['model_name']):
            mlflow.sklearn.log_model(model, "model")
            mlflow.evaluate(
                model=f"runs:/{mlflow.active_run().info.run_id}/model",
                data=pd.DataFrame(data["X_test"], columns=features),
                targets=data["y_test"],
                model_type="classifier"
            )

        return metrics

    def save_results(**kwargs):
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='train_model')
        
        filebuffer = io.BytesIO()
        filebuffer.write(json.dumps(metrics, indent=2).encode())
        filebuffer.seek(0)
        s3_client.upload_fileobj(
            Fileobj=filebuffer,
            Bucket=BUCKET,
            Key=f"metrics/{metrics['model_name']}/metrics.json"
        )

    default_args = {
        "owner": "arsenii_korol",
        "email": 'arsenykingrus@gmail.com',
        "retry": 3,
        "retry_delay": timedelta(minutes=1)
    }

    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",  
        start_date=days_ago(2),
        catchup=False,
        default_args=default_args,
        tags=['mlops']
    )

    with dag:
        task_init = PythonOperator(task_id="init", python_callable=init, dag=dag, op_kwargs={"model_name": model_name})
        task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag)
        task_process_data = PythonOperator(task_id="process_data", python_callable=process_data, dag=dag)
        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
        task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag)

        task_init >> task_get_data >> task_process_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"arsenii_korol_{model_name}", model_name)
