# Advanced House Price Prediction Application

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://housepriceprediction1004.streamlit.app/)

This project is a full-featured web application that predicts house prices using a sophisticated stacking ensemble of machine learning models. It provides not only an accurate price estimate but also an AI-powered explanation of the factors influencing the prediction, all wrapped in a fast, professional, and user-friendly interface built with Streamlit.

---

## 🚀 Live Application

**You can access and use the live application here:**
### [https://housepriceprediction1004.streamlit.app/](https://housepriceprediction1004.streamlit.app/)

---

## ✨ Features

- **Accurate Price Prediction:** Utilizes a stacking ensemble of four different machine learning models (Ridge, Gradient Boosting, XGBoost, LightGBM) with a Lasso meta-model for high accuracy.
- **Explainable AI (XAI):** Integrates SHAP (SHapley Additive exPlanations) to generate a clear waterfall plot, showing exactly which features pushed the price estimate up or down.
- **Interactive UI:** A clean, tabbed interface built with Streamlit allows users to easily input features and analyze results.
- **Market Comparison:** A histogram plot shows how the predicted price compares to the distribution of prices in the original dataset.
- **High-Performance:** Caches all machine learning models on startup for instant predictions and uses on-demand analysis for computationally expensive explanations to ensure a responsive user experience.

---

## 🛠️ Technology Stack

- **Modeling:** Scikit-learn, XGBoost, LightGBM
- **Explainable AI:** SHAP
- **Data Manipulation:** Pandas, NumPy
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Community Cloud

---

## ⚙️ Project Workflow

The model was developed through a rigorous, multi-stage process to ensure accuracy and robustness:

1.  **Data Preprocessing:** The initial Ames Housing dataset was cleaned by handling missing values and converting categorical features into a machine-readable format.

2.  **Advanced Feature Engineering:** To capture complex patterns, new features were created from the base data, including combined features (`TotalSF`), polynomial features (`GrLivArea^2`), and interaction features (`OverallQual * TotalSF`).

3.  **Model Training (Stacking Ensemble):** Instead of relying on a single model, a stacking ensemble was built.
    - **Level 0 (Base Models):** Four diverse models (`RidgeCV`, `GradientBoostingRegressor`, `XGBRegressor`, `LGBMRegressor`) were trained on the engineered features.
    - **Level 1 (Meta-Model):** A final `LassoCV` meta-model was trained on the *predictions* of the base models. This model learns the optimal way to combine the base model predictions, leading to a more accurate and generalized final result.

4.  **Saving Artifacts:** All components needed for deployment—the trained base models, the meta-model, the data scaler, and the SHAP explainer—were saved to disk using `joblib`.

---

## 🖥️ How to Run Locally

Follow these steps to run the application on your own machine.

### Prerequisites

- Python 3.9+
- Git

### 1. Clone the Repository

Open your terminal and clone the repository to your local machine:
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

### 2. Set Up a Virtual Environment (Recommended)

It's best practice to create a virtual environment to manage project dependencies.
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

The `requirements.txt` file contains all the necessary libraries. Install them with a single command:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

Ensure all the `.pkl` artifact files are in the main project directory. Launch the application with:
```bash
streamlit run app.py
```
The application should automatically open in your web browser.
