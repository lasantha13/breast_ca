import streamlit as st
import joblib
import numpy as np
import pandas as pd
# Ensure st.set_page_config() is called first
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded",
)
dark_theme = """
<style>
body { background-color: #202124; color: #ffffff; }
h1, h2, h3, h4, h5, h6 { color: #ffffff; }
.stApp { background-color: #202124; color: #ffffff; }
/* Target the sidebar */
[data-testid="stSidebar"] { background-color: #303134; color: #ffffff; }
/* Adjust colors as needed */
</style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)

#define sidebar
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
    
    input_dict={
       "radius_mean":radius_mean,
         "compactness_mean":compactness_mean,
           "concave_points_mean":concave_points_mean,
                      "compactness_worst":compactness_worst,
                        "concavity_worst":concavity_worst,
                          "concave_points_worst":concave_points_worst
    }
    return(input_dict)
#add prediction
col1,col2 = st.columns([4,1])


def add_prediction(input_df):
    model = joblib.load("artifacts/model.pkl")
    pre_process = joblib.load("artifacts/preprocessor.pkl")

    #input_array = np.array(list(input_data.values())).reshape(1,-1)
    scalded_array = pre_process.transform(input_df)
    return(scalded_array)


def main():
    input_data = add_sidebar()
    
    with col1:
        st.header ("Beast cancer")
    with col1:
        data_list = [{"feature": key, "value": value} for key, value in input_data.items()]
        pred_df = pd.DataFrame(data_list, index=range(len(data_list)))  # Use range(len(data_list)) for unique indices
        st.write(pred_df)
        scaleded =add_prediction(pred_df)
    
     
if __name__ == '__main__':
    main()
