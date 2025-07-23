import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
import xgboost as xgb
import lightgbm as lgb
from app_utils import stacked_prediction # Make sure app_utils.py is in the same folder

print("--- Starting Artifact Creation Process ---")

# 1. Load Data
print("Step 1/4: Loading data...")
X = pd.read_csv('engineered_X_train.csv')
X_test = pd.read_csv('engineered_X_test.csv')
y = pd.read_csv('y_train.csv').values.ravel()
print("Data loaded successfully.")

# 2. Train and Save Base Models
print("\nStep 2/4: Training and saving base models...")
ridge = RidgeCV(alphas=np.logspace(-3, 2, 50))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.02, max_depth=3, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', subsample=0.8, random_state=42)
xgboost = xgb.XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, colsample_bytree=0.7, objective='reg:squarederror', nthread=-1, seed=27, reg_alpha=0.00006)
lightgbm = lgb.LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75, bagging_freq=5, bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7, verbose=-1)
models = {'Ridge': ridge, 'GBR': gbr, 'XGBoost': xgboost, 'LightGBM': lightgbm}

scaler = StandardScaler().fit(pd.concat([X, X_test]))
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')

for name, model in models.items():
    print(f"  - Training {name}...")
    if name == 'Ridge':
        model.fit(scaler.transform(X), y)
    else:
        model.fit(X, y)
    joblib.dump(model, f'base_model_{name}.pkl')
print("Base models, scaler, and columns saved.")

# 3. Train and Save Meta-Model
print("\nStep 3/4: Training and saving the meta-model...")
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], len(models)))

for i, (name, model) in enumerate(models.items()):
    print(f"  - Generating out-of-fold predictions for {name}...")
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
        X_val_fold = X.iloc[val_idx]
        if name == 'Ridge':
            fold_scaler = StandardScaler().fit(X_train_fold)
            model.fit(fold_scaler.transform(X_train_fold), y_train_fold)
            oof_preds[val_idx, i] = model.predict(fold_scaler.transform(X_val_fold))
        else:
            model.fit(X_train_fold, y_train_fold)
            oof_preds[val_idx, i] = model.predict(X_val_fold)

meta_model = LassoCV(cv=5, alphas=np.logspace(-6, -1, 100))
meta_model.fit(oof_preds, y)
joblib.dump(meta_model, 'meta_model.pkl')
print("Meta-model trained and saved.")

# 4. Create and Save SHAP Explainer
print("\nStep 4/4: Creating and saving SHAP explainer...")
# --- OPTIMIZATION: Use a smaller sample size for the background data ---
X_sample = X.sample(30, random_state=42)
print(f"Using a sample of {len(X_sample)} data points for the SHAP explainer background.")

explainer = shap.KernelExplainer(stacked_prediction, X_sample)
joblib.dump(explainer, 'shap_explainer.pkl')
print("SHAP explainer created and saved.")

print("\n--- All artifacts created successfully! ---")
