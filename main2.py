import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import joblib
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_sidebar():
    st.sidebar.header("Input Data")
    columns = ["radius_mean", "compactness_mean", "concave_points_mean",
           "compactness_worst", "concavity_worst", "concave_points_worst"]
    radius_mean = st.sidebar.slider("Radiou mean",0.01,10.00)
    compactness_mean = st.sidebar.slider("Compactness mean",0.01,10.00)
    concave_points_mean= st.sidebar.slider("Concave points mean",0.01,10.00)
    compactness_worst= st.sidebar.slider("Compactness worst",0.01,10.00)
    concavity_worst= st.sidebar.slider("Concavity worst",0.01,10.00)
    concave_points_worst= st.sidebar.slider("Concave points worst",0.01,10.00)
    
    columns = ["radius_mean", "compactness_mean", "concave_points_mean",
           "compactness_worst", "concavity_worst", "concave_points_worst"]
    row = np.array([radius_mean ,compactness_mean ,concave_points_mean ,compactness_worst ,concavity_worst ,concave_points_worst])
    input_df = pd.DataFrame([row], columns =columns)
  
    
    return(input_df)

preprocessor = joblib.load("artifacts/preprocessor.pkl")
# pred_df = add_sidebar()
# st.write(pred_df)

# predict_pipeline=PredictPipeline()
# results=predict_pipeline.predict(pred_df)
# if results[0] == 'B':
#     result = 'Mammogram Results indicate Benign tumor'
#     st.write(result)
    
# else:
#     result = 'Mammogram Results indicate Malignant tumor'
#     st.write(result)
pred_df = add_sidebar()


def prediction():
    
    preprocessor2 = joblib.load("artifacts/preprocessor.pkl")
    pred_df2 =pred_df.T
    st.write(pred_df2) 
    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)
    st.write('Mammogram Results indicate : ')
    if results[0] == 'B':
        result = 'Benign tumor'
        st.subheader(result)
    
    else:
        result = 'Malignant tumor'
        st.subheader(result)
        

trigger = st.button("Prediction",on_click=prediction)

