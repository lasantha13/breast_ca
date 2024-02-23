import streamlit as st
import streamlit_shap as st_shap
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import joblib
import matplotlib.pyplot as plt
import lime
import lime.lime_text 
import lime.lime_tabular

## Route for a home page
st.title("Breast Cancer Prediction")
model = 'xxxxx'
columns = ["radius_mean", "compactness_mean", "concave_points_mean",
           "compactness_worst", "concavity_worst", "concave_points_worst"]
radius_mean = st.slider("Radiou mean",0.01,10.00)
compactness_mean = st.slider("Compactness mean",0.01,10.00)
concave_points_mean= st.slider("Concave points mean",0.01,10.00)
compactness_worst= st.slider("Compactness worst",0.01,10.00)
concavity_worst= st.slider("Concavity worst",0.01,10.00)
concave_points_worst= st.slider("Concave points worst",0.01,10.00)

####### Normal prediction

def get_data_as_data_frame():
  row = np.array([radius_mean ,compactness_mean ,concave_points_mean ,compactness_worst ,concavity_worst ,concave_points_worst])
  X = pd.DataFrame([row], columns =columns)
  pred_df= X
  predict_pipeline=PredictPipeline()
  results=predict_pipeline.predict(pred_df)
  if results[0] == 'B':
    result = 'Mammogram Results indicate Benign tumor'
    st.write(result)
    
  else:
    result = 'Mammogram Results indicate Malignant tumor'
    st.write(result)

trigger = st.button('Predict', on_click=get_data_as_data_frame)



####### USE SHAP ########
model = joblib.load("artifacts/model.pkl")
features = ['radius_mean' ,'compactness_mean' ,'concave_points_mean' ,'compactness_worst' ,'concavity_worst' ,'concave_points_wors']
preprocessor = joblib.load("artifacts/preprocessor.pkl")
def get_shap():
  row = np.array([radius_mean ,compactness_mean ,concave_points_mean ,compactness_worst ,concavity_worst ,concave_points_worst])
  pred_df = pd.DataFrame([row], columns =columns)
 
# preprocessing and prediction both done for input data
  predict_pipeline=PredictPipeline()
  results=predict_pipeline.predict(pred_df)
  if results[0] == 'B':
    result = 'Mammogram Results indicate Benign tumor'
    st.write(result)
    
  else:
    result = 'Mammogram Results indicate Malignant tumor'
    st.write(result)

  #preprocessing for XAI
  data_scaled=preprocessor.transform(pred_df)

  explainer = shap.Explainer(model, data_scaled, feature_names=features)
  shap_values = explainer(data_scaled)
  st.title("Interactive SHAP Explanations for Logistic Regression")
  st.write("Explore the impact of each feature on the model's predictions.")
   # Create SHAP summary plot
  plt.figure(figsize=(12, 8))
  shap.summary_plot(shap_values, data_scaled, feature_names=features)
  plt.tight_layout()
  # Display the plot in Streamlit
  st.pyplot()
trigger = st.button('Shap', on_click=get_shap)

######## LIME ####
def get_lime():
  row = np.array([radius_mean ,compactness_mean ,concave_points_mean ,compactness_worst ,concavity_worst ,concave_points_worst])
  pred_df = pd.DataFrame([row], columns =columns)
  features = ['radius_mean' ,'compactness_mean' ,'concave_points_mean' ,'compactness_worst' ,'concavity_worst' ,'concave_points_wors']
# preprocessing and prediction both done for input data
  predict_pipeline=PredictPipeline()
  results=predict_pipeline.predict(pred_df)
  if results[0] == 'B':
    result = 'Mammogram Results indicate Benign tumor'
    st.write(result)
    
  else:
    result = 'Mammogram Results indicate Malignant tumor'
    st.write(result)

  #preprocessing for XAI
  data_scaled=preprocessor.transform(pred_df)
  explainer  =lime.lime_tabular.LimeTabularExplainer(data_scaled,
                                                   feature_names=features,
                                                   verbose=True,
                                                   mode='classification',
                                                   training_labels='diagnosis',
                                                   class_names=['Benign','Malignant'])
  st.title("Local Explanations with LIME")
  st.write("Select an instance for explanation:")
  exp =explainer.explain_instance(data_scaled[3],model.predict_proba)
  exp.show_in_notebook(show_table=True)
  exp.as_list()
   
trigger = st.button('LIME', on_click=get_lime)


# if __name__=="__main__":
#   app_streamlit()
