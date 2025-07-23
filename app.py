import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="House Price Predictor Pro",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def load_resources():
    """Load all pre-trained models and artifacts once and cache them."""
    artifacts = {}
    required_files = [
        'scaler.pkl', 'meta_model.pkl', 'shap_explainer.pkl', 'model_columns.pkl',
        'base_model_Ridge.pkl', 'base_model_GBR.pkl',
        'base_model_XGBoost.pkl', 'base_model_LightGBM.pkl',
        'y_train.csv'
    ]
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Missing required artifact: '{f}'. Please run the '4_Train_and_Save_Deployment_Artifacts.ipynb' notebook first.")
            return None
        artifacts[f.split('.')[0]] = joblib.load(f) if f.endswith('.pkl') else pd.read_csv(f)
    
    artifacts['base_models'] = {
        'Ridge': artifacts.pop('base_model_Ridge'),
        'GBR': artifacts.pop('base_model_GBR'),
        'XGBoost': artifacts.pop('base_model_XGBoost'),
        'LightGBM': artifacts.pop('base_model_LightGBM')
    }
    return artifacts

resources = load_resources()

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

def predict_price(input_df, resources):
    """Performs the FAST price prediction using the pre-loaded models."""
    if not resources: return None
    
    final_df = input_df.reindex(columns=resources['model_columns'], fill_value=0)
    
    base_predictions = np.zeros((1, 4))
    for i, (name, model) in enumerate(resources['base_models'].items()):
        if name == 'Ridge':
            data_scaled = resources['scaler'].transform(final_df)
            base_predictions[0, i] = model.predict(data_scaled)
        else:
            base_predictions[0, i] = model.predict(final_df)

    final_log_prediction = resources['meta_model'].predict(base_predictions)
    final_price = np.expm1(final_log_prediction[0])
    
    return final_price

@st.cache_data
def get_shap_explanation(_input_df_tuple, _resources):
    """
    Performs the SLOW SHAP analysis. Caching this function is key.
    Streamlit can't hash DataFrames directly, so we pass a tuple of its values.
    """
    input_df = pd.DataFrame([_input_df_tuple], columns=resources['model_columns'])
    shap_values = resources['shap_explainer'].shap_values(input_df)
    
    explanation = shap.Explanation(
        values=shap_values[0,:],
        base_values=resources['shap_explainer'].expected_value,
        data=input_df.iloc[0,:],
        feature_names=resources['model_columns']
    )
    return explanation

st.title('📈 Professional House Price Predictor')

if not resources:
    st.stop()

tab1, tab2, tab3 = st.tabs(["🏠 Price Predictor", "📊 Prediction Analysis", "ℹ️ About"])

with tab1:
    st.header("Enter House Features")
    
    with st.form(key='prediction_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Overall Quality")
            overall_qual = st.slider('Overall Material & Finish', 1, 10, 7)
            exter_qual = st.selectbox('Exterior Quality', ['Ex', 'Gd', 'TA', 'Fa'], index=1)
            kitchen_qual = st.selectbox('Kitchen Quality', ['Ex', 'Gd', 'TA', 'Fa'], index=1)
        with col2:
            st.subheader("Size & Area (sqft)")
            gr_liv_area = st.number_input('Above Ground Living Area', 500, 5000, 1710, 50)
            total_bsmt_sf = st.number_input('Total Basement Area', 0, 6000, 856, 50)
            garage_area = st.number_input('Garage Area', 0, 1500, 548, 20)
        with col3:
            st.subheader("Age & Location")
            year_built = st.number_input('Year Built', 1870, 2022, 2003)
            neighborhood = st.selectbox('Neighborhood', ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber', 'IDOTRR', 'ClearCr', 'StoneBr', 'SWISU', 'MeadowV', 'Blmngtn', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'])
            full_bath = st.selectbox('Full Bathrooms', [1, 2, 3, 4], index=1)

        submit_button = st.form_submit_button(label='Predict Price', use_container_width=True)

    if submit_button:
        user_input_dict = {
            'OverallQual': overall_qual, f'ExterQual_{exter_qual}': 1, f'KitchenQual_{kitchen_qual}': 1,
            'GrLivArea': gr_liv_area, 'TotalBsmtSF': total_bsmt_sf, 'GarageArea': garage_area,
            'YearBuilt': year_built, f'Neighborhood_{neighborhood}': 1, 'FullBath': full_bath,
            '1stFlrSF': gr_liv_area / 2, '2ndFlrSF': gr_liv_area / 2,
        }
        
        featured_df = create_features(pd.DataFrame([user_input_dict]))
        final_df = featured_df.reindex(columns=resources['model_columns'], fill_value=0)
        
        predicted_price = predict_price(final_df, resources)
        
        st.session_state['predicted_price'] = predicted_price
        st.session_state['final_df'] = final_df

    if 'predicted_price' in st.session_state:
        st.markdown("---")
        st.header("Prediction Result")
        st.metric(label="Estimated House Price", value=f"${st.session_state['predicted_price']:,.0f}")
        st.info("View the 'Prediction Analysis' tab for a detailed breakdown of this estimate.", icon="📊")

with tab2:
    st.header("Detailed Prediction Analysis")
    if 'final_df' not in st.session_state:
        st.info("Please enter features and click 'Predict Price' on the first tab to generate a prediction.")
    else:
        st.markdown("Click the button below to run a detailed analysis of the prediction. This may take a moment.")
        
        if st.button("Run Detailed Analysis", type="primary"):
            st.session_state['analysis_clicked'] = True

        if st.session_state.get('analysis_clicked', False):
            with st.spinner('Calculating feature impacts...'):
                df_tuple = tuple(st.session_state['final_df'].iloc[0])
                explanation = get_shap_explanation(df_tuple, resources)

            st.subheader("Feature Contribution Breakdown (Waterfall Plot)")
            st.markdown("This chart shows how each feature contributes to the final price, starting from the average market price.")
            
            fig_waterfall, ax_waterfall = plt.subplots()
            shap.waterfall_plot(explanation, max_display=15, show=False)
            st.pyplot(fig_waterfall, bbox_inches='tight')
            plt.close(fig_waterfall)

            st.markdown("---")
            st.subheader("How does this house compare to the market?")
            
            fig_hist, ax_hist = plt.subplots()
            prices = np.expm1(resources['y_train']['SalePrice'])
            ax_hist.hist(prices, bins=60, alpha=0.7, label='Market Distribution', color='skyblue')
            ax_hist.axvline(st.session_state['predicted_price'], color='red', linestyle='--', lw=2, label='Your House Estimate')
            ax_hist.set_title("Prediction vs. Market Prices")
            ax_hist.set_xlabel("Price ($)")
            ax_hist.set_ylabel("Number of Houses")
            ax_hist.legend()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

with tab3:
    st.header("About This App")
    st.markdown("""
    This tool predicts house prices using a sophisticated stacking ensemble of machine learning models. 
    It analyzes property details to provide an accurate, data-driven estimate of a home's market value.
    """)
    st.subheader("Key Features Used by the Model")
    st.markdown("""
    - **Overall Quality:** Rates the overall material and finish of the house.
    - **Living Area:** The total above-ground living space in square feet.
    - **Basement & Garage Area:** The total square footage of key areas.
    - **Year Built:** The original construction date.
    - **Exterior & Kitchen Quality:** Rates the quality of important rooms.
    - **Neighborhood:** The physical location of the property within Ames, Iowa.
    """)
    st.subheader("Technology Stack")
    st.markdown("""
    - **Modeling:** Scikit-learn, XGBoost, LightGBM
    - **Explainable AI:** SHAP (SHapley Additive exPlanations)
    - **Web Framework:** Streamlit
    """)