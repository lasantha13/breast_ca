import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import joblib
import shap
import streamlit_shap as st_shap
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import shapash as sh
from shapash.explainer.smart_explainer import SmartExplainer
#from lime.lime_tabular import LimeTabularExplainer
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded",
)

def add_sidebar():
    st.sidebar.header("Breast Mass: Risk Prediction")
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
    st.write("Predicted Result : ")
    st.write(pred_df2) 
    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)
    st.subheader('Mammogram Results indicate : ')
    if results[0] == 'B':
        result = 'Benign tumor'
        st.success(result)
    
    else:
        result = 'Malignant tumor'
        st.error(result)
       


features = ['radius_mean' ,'compactness_mean' ,'concave_points_mean' ,'compactness_worst' ,'concavity_worst' ,'concave_points_wors']
train_data= pd.read_csv('New/x2_train.csv')

def scalded_value():
    #get train data 
    scaler = StandardScaler()
    train_scalded = scaler.fit_transform(train_data)
    input_scalded  = scaler.transform(pred_df)
    return(train_scalded,input_scalded)
# def scalded_input():
#     scaler = StandardScaler()
#     train_scalded = scaler.fit_transform(train_data)
#     input_scalded  = scaler.transform(pred_df)
#     return(train_scalded,input_scalded)
def lime_exp():
    train_scalded,inputs_scalded = scalded_value()
    explainer  =lime.lime_tabular.LimeTabularExplainer(train_scalded,
                                                   feature_names=features,
                                                   verbose=True,                                                   
                                                   mode='classification',
                                                   training_labels='diagnosis',
                                                   class_names=['Benign','Malignant'])
    
    model = joblib.load("artifacts/model.pkl")
    exp =explainer.explain_instance(train_scalded[6],model.predict_proba)
   #exp = explainer.explain_instance(pred_df.iloc[instance_index], model.predict_proba, num_features=len(feature_names))
    st.write("LIME Visuals--------")
    exp_image = exp.as_pyplot_figure()
    exp_image.canvas.draw()
    image_data = np.frombuffer(exp_image.canvas.tostring_rgb(), dtype=np.uint8)
    image_data = image_data.reshape(exp_image.canvas.get_width_height()[::-1] + (3,))

    st.image(image_data, caption='LIME Explanation', use_column_width=True)
def shap_plot():
    st.subheader("Prediction model - Feature Importance")
    col1,col2 =st.columns([1,1])
    with col1:
            train_scalded,inputs_scalded = scalded_value()
            model = joblib.load("artifacts/model.pkl")
            explainer = shap.Explainer(model, train_scalded, feature_names=features)
            shap_values = explainer(train_scalded)
            fig, ax = plt.subplots(figsize=(3,2))
            shap.summary_plot(shap_values, train_scalded, plot_type="bar",feature_names=features)
            st.pyplot(fig)
    with col2:
            train_scalded,inputs_scalded = scalded_value()
            model = joblib.load("artifacts/model.pkl")
            explainer = shap.Explainer(model, train_scalded, feature_names=features)
            shap_values = explainer(train_scalded)
            fig, ax = plt.subplots(figsize=(3,2))
            shap.summary_plot(shap_values, train_scalded, feature_names=features)
            st.pyplot(fig)
            

def shap_waterfallplot():
    cotainer1 =st.container()
    with cotainer1:
        col1,col2 =st.columns([2,3])
        with col1:
            prediction() 
        
        with col2:
            # Load the model
            # Create explainer object using the pre-scaled data
            # Calculate Shap values
            # Create the waterfall plot using shap.waterfall
            train_scalded,inputs_scalded = scalded_value()
            model = joblib.load("artifacts/model.pkl")
            fig, ax = plt.subplots(figsize=(10, 5))  # Adjust figure size as needed

            explainer = shap.Explainer(model, train_scalded, feature_names=features)
            shap_values2 = explainer(inputs_scalded)
            # Display the plot in Streamlit
            shap.waterfall_plot(shap_values2[0])
            st.pyplot(fig)
    cotainer2 =st.container()
    with cotainer2:
        st.write("_____________________________________________________________________________________________________________")
                  
            
##SHAPASH
def shapash_create():
    scalded_values = scalded_value()
    model1 = joblib.load("artifacts/model.pkl")
    # Declare SmartExplainer
    xpl = SmartExplainer(model1)

    # Compile SmartExplainer with your model
    xpl.compile(x=scalded_values, model=model1)

    

    
    # Streamlit app
    st.title('Shapash Report')

    # Select the index you want to explain
    index_to_explain = st.selectbox('Select index to explain',scalded_values.index)

    # Display the local explanation
    local_plot = xpl.plot.local_plot(index=index_to_explain, show=False)
    st.pyplot(local_plot)
shap_plot()
shap_waterfallplot()
st.sidebar.button("Prediction",on_click=prediction)
st.sidebar.button("SHAP Graph",on_click=shap_plot)
st.sidebar.button("LIME Graph",on_click=lime_exp)