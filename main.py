import streamlit as st

# Ensure st.set_page_config() is called first
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.write("test")

with st.container():
  st.title("Breast Cancer Prediction")
  st.write("Update the mammogram values")

col1,col2 = st.columns([4,1])

with col1:
   st.write("column 1")
with col2:
   st.write("column 2")

if __name__ == '__main__':
    main()
