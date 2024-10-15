


# Создадим parent run.
with mlflow.start_run(run_name="parent_run", experiment_id = experiment_id, description = "parent") as parent_run:
    for model_name in models.keys():
        # Запустим child run на каждую модель.
        with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, nested=True) as child_run:
            model = models[model_name]
            
            # Обучим модель.
            model.fit(pd.DataFrame(X_train), y_train)
        
            # Сделаем предсказание.
            prediction = model.predict(X_val)
        
            # Создадим валидационный датасет.
            eval_df = X_val.copy()
            eval_df["target"] = y_val
        
            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, "logreg", signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )

