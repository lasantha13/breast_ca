import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import joblib
import shap
import streamlit_shap as st_shap
import matplotlib.pyplot as plt

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

pred_df = add_sidebar()

def prediction():
    pred_df2 =pred_df.T
    st.write("Input Values")
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
       


features = ['radius_mean' ,'compactness_mean' ,'concave_points_mean' ,'compactness_worst' ,'concavity_worst' ,'concave_points_wors']
train_data= pd.read_csv('New/x2_train.csv')

def scalded_value():
    #get train data 
    scaler = StandardScaler()
    train_scalded = scaler.fit_transform(train_data)
    return train_scalded

def shap_plot():
    col1,col2 =st.columns([1,2])
    with col1:
        prediction()
    with col2:
        scalded_values = scalded_value()
        model = joblib.load("artifacts/model.pkl")
        explainer = shap.Explainer(model, scalded_values, feature_names=features)
        shap_values = explainer(scalded_values)
        fig, ax = plt.subplots(figsize=(3,2))
        shap.summary_plot(shap_values, scalded_values, feature_names=features)
        st.pyplot(fig)




container1 = st.container()
container2 = st.container()



st.sidebar.button("Prediction",on_click=prediction)
st.sidebar.button("SHAP Graph",on_click=shap_plot)