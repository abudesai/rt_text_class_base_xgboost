
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

from xgboost import XGBClassifier


model_fname = "model.save"
MODEL_NAME = "text_class_base_xgboost"


class Classifier(): 
    
    def __init__(self, booster="gbtree", eta=0.3, gamma=0.0, max_depth=6, **kwargs) -> None:
        self.booster = booster
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.model = self.build_model(**kwargs)     
        
        
    def build_model(self, **kwargs): 
        model = XGBClassifier(
            booster = self.booster,
            eta = self.eta,
            gamma = self.gamma,
            max_depth = self.max_depth,
            **kwargs)
        return model
    
    
    def fit(self, train_X, train_y):        
        self.model.fit(train_X, train_y)            
        
    
    def predict(self, X, verbose=False): 
        preds = self.model.predict(X)
        return preds 
    
    
    def predict_proba(self, X, verbose=False): 
        preds = self.model.predict_proba(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))
        


    @classmethod
    def load(cls, model_path):         
        rfclassifier = joblib.load(os.path.join(model_path, model_fname))
        # print("where the load function is getting the model from: "+ os.path.join(model_path, model_fname))        
        return rfclassifier


def save_model(model, model_path):
    # print(os.path.join(model_path, model_fname))
    joblib.dump(model, os.path.join(model_path, model_fname)) #this one works
    # print("where the save_model function is saving the model to: " + os.path.join(model_path, model_fname))
    

def load_model(model_path): 
    model = joblib.load(os.path.join(model_path, model_fname))   
    try: 
        model = joblib.load(os.path.join(model_path, model_fname))   
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model


