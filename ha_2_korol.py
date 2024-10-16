import mlflow
import os
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = ['age', 'fnlwgt', 'education-num',
            'capital-gain', 'capital-loss', 'hours-per-week']

models = {
    "logistic_regression": LogisticRegression(),
    "gradient_boosting": GradientBoostingClassifier(),
    "decision_tree": DecisionTreeClassifier()}

exp_id = mlflow.create_experiment(name="Arsenii Korol")

dataset = fetch_openml(name='adult', version=2, as_frame=True)

# Признаки (features) и целевая переменная (target)
X = dataset['data']
y = dataset['target']

# Кодирование целевой переменной в 0 и 1
y = y.map({'<=50K': 0, '>50K': 1})

# Объединение данных признаков и целевой переменной в один DataFrame
data = pd.concat([X, y.rename('target')], axis=1)

X, y = data[features], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)


with mlflow.start_run(run_name="senyafeelsgood", experiment_id=exp_id, description="parent") as parent_run:
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, experiment_id=exp_id, nested=True) as child_run:
            model.fit(X_train, y_train)
            
            prediction = model.predict(X_test)
            
            r2 = r2_score(y_test, prediction)
            mse = mean_squared_error(y_test, prediction)
            rmse = mean_squared_error(y_test, prediction, squared=False)
            mae = mean_absolute_error(y_test, prediction)
            max_err = max_error(y_test, prediction)
            example_count = len(y_test)
            mean_on_target = y_test.astype(float).mean()

            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mean_squared_error", mse)
            mlflow.log_metric("root_mean_squared_error", rmse)
            mlflow.log_metric("mean_absolute_error", mae)
            mlflow.log_metric("max_error", max_err)
            mlflow.log_metric("example_count", example_count)
            mlflow.log_metric("mean_on_target", mean_on_target)
            mlflow.log_metric("score", r2)

            eval_df = X_test.copy()
            eval_df["target"] = y_test

            signature = infer_signature(X_test, prediction)
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path=model_name, 
                signature=signature, 
                registered_model_name=f"{model_name}_reg_model"
            )

            mlflow.evaluate(
                model=f"runs:/{child_run.info.run_id}/{model_name}",
                data=eval_df,
                targets="target",
                model_type="classifier",
                evaluators=["default"],
            )
            
            print(f"Finished logging for model: {model_name}")