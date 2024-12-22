import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from joblib import parallel_backend

# Load the dataset
file_path = "G:\\Big DATA\\Semester 4\\Major Project\\fp\\Cleaned_Firegrowth_groups.csv"
data = pd.read_csv(file_path)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Train and evaluate a given model."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Output results
    print(f"\n{model_name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    return rmse, mse, mae, r2


def air_quality_impact_prediction(data):
    # Create a target variable for air quality degradation (example)
    data['air_quality_index'] = data['firearea'] * 0.1 + np.random.normal(0, 5, len(data))  # Simulated relationship

    features = ['firearea', 'fwi', 'ws', 'rh']  # Example features affecting air quality
    X = data[features]
    y = data['air_quality_index']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_rmse, rf_mse, rf_mae, rf_r2 = evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest")

    # XGBoost
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_rmse, xgb_mse, xgb_mae, xgb_r2 = evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_train, y_test, "XGBoost")

    # SVM
    svm_model = SVR(kernel='rbf')
    svm_rmse, svm_mse, svm_mae, svm_r2 = evaluate_model(svm_model, X_train_scaled, X_test_scaled, y_train, y_test, "SVM")

    # Comparison of models
    print("\nModel Comparison:")
    print(f"Random Forest: RMSE={rf_rmse:.2f}, R²={rf_r2:.4f}")
    print(f"XGBoost: RMSE={xgb_rmse:.2f}, R²={xgb_r2:.4f}")
    print(f"SVM: RMSE={svm_rmse:.2f}, R²={svm_r2:.4f}")


# Run the air quality impact prediction
air_quality_impact_prediction(data)




def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Train and evaluate a given model."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Output results
    print(f"{model_name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    return model, rmse, mse, mae, r2


def fire_area_prediction(data, n_samples=10000):
    # Use a subset of data for faster processing
    if len(data) > n_samples:
        data = data.sample(n=n_samples, random_state=42)

    # Split features and target variable
    X = data.drop(['DOB', 'firearea'], axis=1)
    y = data['firearea']

    # Feature selection
    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold='median')
    X_selected = selector.fit_transform(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Random Forest with Hyperparameter Tuning
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    with parallel_backend('threading', n_jobs=-1):
        rf_model = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, n_iter=10, cv=3, n_jobs=-1,
                                      verbose=2)
        rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)

    print("Random Forest Results:")
    print(f"RMSE: {rf_rmse:.2f}")
    print(f"MSE: {rf_mse:.2f}")
    print(f"MAE: {rf_mae:.2f}")
    print(f"R2: {rf_r2:.4f}")
    print("Best parameters:", rf_model.best_params_)

    # XGBoost
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        eval_metric='rmse'
    )
    xgb_model, xgb_rmse, xgb_mse, xgb_mae, xgb_r2 = evaluate_model(xgb_model, X_train, X_test, y_train, y_test,
                                                                   "XGBoost")

    # SVM
    svm_model = SVR(kernel='rbf')
    svm_model, svm_rmse, svm_mse, svm_mae, svm_r2 = evaluate_model(svm_model, X_train, X_test, y_train, y_test, "SVM")

    print("\nComparison of Models:")
    print(f"Random Forest: RMSE={rf_rmse:.2f}, R2={rf_r2:.4f}")
    print(f"XGBoost: RMSE={xgb_rmse:.2f}, R2={xgb_r2:.4f}")
    print(f"SVM: RMSE={svm_rmse:.2f}, R2={svm_r2:.4f}")






import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "G:\\Big DATA\\Semester 4\\Major Project\\fp\\Cleaned_Firegrowth_groups.csv"
data = pd.read_csv(file_path)


def air_quality_impact_prediction(data):
    # Create a target variable for air quality degradation (example)
    data['air_quality_index'] = data['firearea'] * 0.1 + np.random.normal(0, 5, len(data))  # Simulated relationship

    features = ['firearea', 'fwi', 'ws', 'rh']  # Example features affecting air quality
    X = data[features]
    y = data['air_quality_index']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    predictions = rf_model.predict(X_test_scaled)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nAir Quality Impact Prediction (Random Forest):")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")


# Run the air quality impact prediction model
air_quality_impact_prediction(data)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from joblib import parallel_backend

def fire_area_prediction(data, n_samples=10000):
    # Use a subset of data for faster processing
    if len(data) > n_samples:
        data = data.sample(n=n_samples, random_state=42)

    # Split features and target variable
    X = data.drop(['DOB', 'firearea'], axis=1)
    y = data['firearea']

    # Feature selection
    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold='median')
    X_selected = selector.fit_transform(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    with parallel_backend('threading', n_jobs=-1):
        rf_model = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_grid, n_iter=10, cv=3, n_jobs=-1, verbose=2)
        rf_model.fit(X_train, y_train)

    # Predictions
    predictions = rf_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Output results
    print("Fire Area Prediction (Random Forest):")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    print("Best parameters:", rf_model.best_params_)

# Load data and call the function
file_path = "G:\\Big DATA\\Semester 4\\Major Project\\fp\\Cleaned_Firegrowth_groups.csv"
data = pd.read_csv(file_path)
fire_area_prediction(data)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from joblib import parallel_backend


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Train and evaluate a given model"""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Output results
    print(f"{model_name} Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    return model, rmse, mse, mae, r2


def fire_area_prediction(data, n_samples=10000):
    # Use a subset of data for faster processing
    if len(data) > n_samples:
        data = data.sample(n=n_samples, random_state=42)

    # Split features and target variable
    X = data.drop(['DOB', 'firearea'], axis=1)
    y = data['firearea']

    # Feature selection
    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold='median')
    X_selected = selector.fit_transform(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Random Forest with Hyperparameter Tuning
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    with parallel_backend('threading', n_jobs=-1):
        rf_model = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, n_iter=10, cv=3, n_jobs=-1,
                                      verbose=2)
        rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)

    print("Random Forest Results:")
    print(f"RMSE: {rf_rmse:.2f}")
    print(f"MSE: {rf_mse:.2f}")
    print(f"MAE: {rf_mae:.2f}")
    print(f"R2: {rf_r2:.4f}")
    print("Best parameters:", rf_model.best_params_)

    # XGBoost
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        eval_metric='rmse'
    )
    xgb_model, xgb_rmse, xgb_mse, xgb_mae, xgb_r2 = evaluate_model(xgb_model, X_train, X_test, y_train, y_test,
                                                                   "XGBoost")

    # SVM
    svm_model = SVR(kernel='rbf')
    svm_model, svm_rmse, svm_mse, svm_mae, svm_r2 = evaluate_model(svm_model, X_train, X_test, y_train, y_test, "SVM")

    print("\nComparison of Models:")
    print(f"Random Forest: RMSE={rf_rmse:.2f}, R2={rf_r2:.4f}")
    print(f"XGBoost: RMSE={xgb_rmse:.2f}, R2={xgb_r2:.4f}")
    print(f"SVM: RMSE={svm_rmse:.2f}, R2={svm_r2:.4f}")


# Load data and call the function
file_path = "G:\\Big DATA\\Semester 4\\Major Project\\fp\\Cleaned_Firegrowth_groups.csv"
data = pd.read_csv(file_path)
fire_area_prediction(data)



