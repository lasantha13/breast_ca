import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        radius_mean :int,
        concave_points_mean:int,
        compactness_mean :int,
        concave_points_worst:int,
        concavity_worst:int,
        compactness_worst:int):

        self.radius_mean = radius_mean

        self.concave_points_mean = concave_points_mean

        self.compactness_mean = compactness_mean

        self.concave_points_worst = concave_points_worst

        self.concavity_worst = concavity_worst

        self.compactness_worst = compactness_worst

        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "radius_mean": [self.radius_mean],
                "compactness_mean": [self.compactness_mean],
                "concave_points_mean": [self.concave_points_mean],
                "compactness_worst": [self.compactness_worst],
                "concavity_worst": [self.concavity_worst],
                "concave_points_worst": [self.concave_points_worst]
                

                        
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
