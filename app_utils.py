import pandas as pd
import numpy as np
import joblib

def create_features(input_df):
    """Replicates the feature engineering steps on a new data point."""
    df = input_df.copy()
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    poly_features = ['GrLivArea', 'TotalSF', 'TotalBsmtSF', 'OverallQual', 'GarageArea', 'YearBuilt']
    for feature in poly_features:
        if feature in df.columns:
            df[f'{feature}_sq'] = df[feature] ** 2
            df[f'{feature}_cub'] = df[feature] ** 3
    if 'OverallQual' in df.columns and 'TotalSF' in df.columns:
        df['Qual_x_TotalSF'] = df['OverallQual'] * df['TotalSF']
    return df

def stacked_prediction(data):
    """
    This function is required by the SHAP explainer. It replicates the full prediction
    pipeline by loading all the necessary models and processing the data.
    """
    try:
        model_cols = joblib.load('model_columns.pkl')
        scaler = joblib.load('scaler.pkl')
        meta_model = joblib.load('meta_model.pkl')
        base_model_names = ['Ridge', 'GBR', 'XGBoost', 'LightGBM']
    except FileNotFoundError:
        return np.zeros(data.shape[0])

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=model_cols)

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.fillna(0)

    base_predictions = np.zeros((data.shape[0], len(base_model_names)))
    for i, name in enumerate(base_model_names):
        model = joblib.load(f'base_model_{name}.pkl')
        if name == 'Ridge':
            data_scaled = scaler.transform(data)
            base_predictions[:, i] = model.predict(data_scaled)
        else:
            base_predictions[:, i] = model.predict(data)
            
    return meta_model.predict(base_predictions)