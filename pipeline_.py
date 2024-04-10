from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import numpy as np

class AgeImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Age']=imputer.fit_transform(X[['Age']])
        return X
    
class FeatureEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
      # Embarked can have 4 values: C, Q, S, N, so create a new column for each value, put 1 if the value is present, 0 otherwise
      X["C"] = (X["Embarked"] == "C").astype(float)
      X["Q"] = (X["Embarked"] == "Q").astype(float)
      X["S"] = (X["Embarked"] == "S").astype(float)
      X["N"] = (X["Embarked"] == "N").astype(float)

      #Sex can have 2 values: male/female
      X["male"] = (X["Sex"]== "male").astype(float)
      X["female"] = (X["Sex"]== "female").astype(float)
      return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

      return X.drop(["Embarked","Name","Ticket","Cabin","Sex","N"], axis=1, errors="ignore")
    

def preprocess_data(data):
    pipeline=Pipeline([("ageimputer", AgeImputer()),
                   ("featureencoder",FeatureEncoder()),
                   ("featuredropper", FeatureDropper())])
    
    return pipeline.transform(data)

def load_model():
    return joblib.load('final_clf.pkl')

def predict_survival(data):
    model = load_model()
    preprocessed_data = preprocess_data(data)
    preprocessed_data = preprocessed_data.fillna(method='ffill')
    preprocessed_data_final = preprocessed_data.to_numpy()  

    return model.predict(preprocessed_data_final)[0]
    

#create a sample datafram to test the pipeline
data = pd.DataFrame({'PassengerId': 898, 'Pclass': 1 , 'Name': "Snyder, Mrs. John Pillsbury (Nelle Stevenson)",
                     'Sex': "female", 'Age': 30, 'SibSp': 0, 'Parch': 0,
                        'Ticket': "21228", 'Fare': 7.6292, 'Cabin': "B45", 'Embarked': "Q"}, index=[0])

print(predict_survival(data))