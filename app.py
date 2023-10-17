from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            radius_mean=request.form.get('radius_mean'),
            concave_points_mean=request.form.get('concave_points_mean'),
            compactness_mean=request.form.get('compactness_mean'),
            concave_points_worst=request.form.get('concave_points_worst'),
            concavity_worst=request.form.get('concavity_worst'),
            compactness_worst=float(request.form.get('compactness_worst')),
            

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        if results[0] == 'B':
            result = 'Mammogram Results indicate Benign tumor'
        else:
            result = 'Mammogram Results indicate Malignant tumor'
        return render_template('home.html',results=result)
    

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0")        

