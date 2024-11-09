from data_loader import load_wine_data, load_drug_data, load_iris_data, load_titanic_data
from preprocessing import create_wine_preprocessor, create_drug_preprocessor, create_iris_preprocessor, create_titanic_preprocessor
from model_tuning import (create_rf_pipeline, create_lr_pipeline, create_xg_pipeline, 
                          perform_grid_search, perform_random_search, perform_bayesian_search, 
                          get_rf_search_spaces, get_lr_search_spaces, get_xg_search_spaces)

def main():
    # Load data
    X_wine, y_wine = load_wine_data()
    X_drug, y_drug = load_drug_data()
    X_iris, y_iris = load_iris_data()
    X_titanic_train, y_titanic_train, _, _ = load_titanic_data()

    # Preprocessing
    wine_preprocessor = create_wine_preprocessor()
    drug_preprocessor = create_drug_preprocessor()
    iris_preprocessor = create_iris_preprocessor()
    titanic_preprocessor = create_titanic_preprocessor()

    # Model Pipelines
    rf_pipeline = create_rf_pipeline(wine_preprocessor)
    lr_pipeline = create_lr_pipeline(drug_preprocessor)
    xg_pipeline = create_xg_pipeline(iris_preprocessor)

    # Get params
    rf_param_grid, rf_search_spaces = get_rf_search_spaces()
    lr_param_grid, lr_search_spaces = get_lr_search_spaces()
    xg_param_grid, xg_search_spaces = get_xg_search_spaces()

    # Perform Searches
    print("Random Forest Tuning:")
    rf_grid_search = perform_grid_search(rf_pipeline, rf_param_grid, X_wine, y_wine)
    rf_random_search = perform_random_search(rf_pipeline, rf_param_grid, X_wine, y_wine)
    rf_bayesian_search = perform_bayesian_search(rf_pipeline, rf_search_spaces, X_wine, y_wine)
    print("Best RF Grid Search Params:", rf_grid_search.best_params_)
    print("Best RF Random Search Params:", rf_random_search.best_params_)
    print("Best RF Bayesian Search Params:", rf_bayesian_search.best_params_)

    print("\nLogistic Regression Tuning:")
    lr_grid_search = perform_grid_search(lr_pipeline, lr_param_grid, X_drug, y_drug)
    lr_random_search = perform_random_search(lr_pipeline, lr_param_grid, X_drug, y_drug)
    lr_bayesian_search = perform_bayesian_search(lr_pipeline, lr_search_spaces, X_drug, y_drug)
    print("Best LR Grid Search Params:", lr_grid_search.best_params_)
    print("Best LR Random Search Params:", lr_random_search.best_params_)
    print("Best LR Bayesian Search Params:", lr_bayesian_search.best_params_)

    print("\nXGBoost Tuning:")
    xg_grid_search = perform_grid_search(xg_pipeline, xg_param_grid, X_iris, y_iris)
    xg_random_search = perform_random_search(xg_pipeline, xg_param_grid, X_iris, y_iris)
    xg_bayesian_search = perform_bayesian_search(xg_pipeline, xg_search_spaces, X_iris, y_iris)
    print("Best XG Grid Search Params:", xg_grid_search.best_params_)
    print("Best XG Random Search Params:", xg_random_search.best_params_)
    print("Best XG Bayesian Search Params:", xg_bayesian_search.best_params_)

if __name__ == "__main__":
    main()
