from data_loader import load_wine_data, load_drug_data, load_iris_data, load_titanic_data
from preprocessing import create_wine_preprocessor, create_drug_preprocessor, create_iris_preprocessor, create_titanic_preprocessor
from model_tuning import (create_rf_pipeline, create_lr_pipeline, create_xg_pipeline, 
                          perform_grid_search, perform_random_search, perform_bayesian_search, 
                          get_rf_search_spaces, get_lr_search_spaces, get_xg_search_spaces)

def main():
    X_wine, y_wine = load_wine_data()
    X_drug, y_drug = load_drug_data()
    X_iris, y_iris = load_iris_data()
    X_titanic_train, y_titanic_train, _, _ = load_titanic_data()

    wine_preprocessor = create_wine_preprocessor()
    drug_preprocessor = create_drug_preprocessor()
    iris_preprocessor = create_iris_preprocessor()
    titanic_preprocessor = create_titanic_preprocessor()

    rf_param_grid, rf_search_spaces = get_rf_search_spaces()
    lr_param_grid, lr_search_spaces = get_lr_search_spaces()
    xg_param_grid, xg_search_spaces = get_xg_search_spaces()

    datasets = [
        ('Wine', X_wine, y_wine, wine_preprocessor),
        ('Drug', X_drug, y_drug, drug_preprocessor),
        ('Iris', X_iris, y_iris, iris_preprocessor),
        ('Titanic', X_titanic_train, y_titanic_train, titanic_preprocessor)
    ]

    models = [
        ('Random Forest', create_rf_pipeline, rf_param_grid, rf_search_spaces),
        ('Logistic Regression', create_lr_pipeline, lr_param_grid, lr_search_spaces),
        ('XGBoost', create_xg_pipeline, xg_param_grid, xg_search_spaces)
    ]

    for dataset_name, X, y, preprocessor in datasets:
        print(f"\nDataset: {dataset_name}")
        for model_name, create_pipeline_func, param_grid, search_spaces in models:
            print('#'*50)
            print(f"\n{model_name} Tuning on {dataset_name}:")
            print('#'*50)
            pipeline = create_pipeline_func(preprocessor)
            try:
                grid_search = perform_grid_search(pipeline, param_grid, X, y)
                random_search = perform_random_search(pipeline, param_grid, X, y)
                bayesian_search = perform_bayesian_search(pipeline, search_spaces, X, y)
                print(f"Best {model_name} params for Grid Search:", grid_search.best_params_)
                print(f"Best {model_name} params for Random Search:", random_search.best_params_)
                print(f"Best {model_name} params for Bayesian Search:", bayesian_search.best_params_)
            except Exception as e:
                print(f":'( error in {model_name} on {dataset_name}: {e}")

if __name__ == "__main__":
    main()
