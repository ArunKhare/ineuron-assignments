import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from operator import itemgetter
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import streamlit as st
from pathlib import Path
from dataclasses import dataclass

@dataclass
class model_params:
    models: dict
    params: dict

@dataclass
class split_data:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.result = []
    
    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
            
            self.result.append(
                {
                    'grid': grid_search,
                    'classifier': grid_search.best_estimator_,
                    'best score': grid_search.best_score_,
                    'best params': grid_search.best_params_,
                    'cv': grid_search.cv
                }
            )
        print('Done.')
    
    def best_result(self):
        result = sorted(self.result, key=itemgetter('best score'), reverse=True)
        grid = result[0]['grid']
        joblib.dump(grid, 'classifier.pickle')
        return result
    
    def predict_proba_with_threshold(self, X, threshold):
        best_classifier = self.result[0]['classifier']
        y_proba = best_classifier.predict_proba(X)
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        return y_pred, y_proba

class DataPreprocessingApp:
    threshold_values = np.arange(0.1, 0.9, 0.1)
    missing_values_threshold = .2
    def __init__(self):
        self.is_na = False
        self.model_list = [LogisticRegression, RandomForestClassifier, DecisionTreeClassifier,GradientBoostingClassifier]
        self.root_dir = os.path.join(os.getcwd())
    def get_data(self):
        path = Path(os.path.join(self.root_dir,"train.csv"))
        df = pd.read_csv(path)
        return df
    
    def preprocessing(self, df):
        df.drop(columns=['Loan_ID'], inplace=True)
        
        data_na = (df.isnull().sum() / len(df)) 
        data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
        
        if not data_na.empty:
            self.is_na = True
            assert self.is_na == True
        
        if (data_na >= DataPreprocessingApp.missing_values_threshold).any(): 
            # na_column_to_drop = list(data_na[data_na >= DataPreprocessingApp.missing_values_threshold].index)
            # df.drop(columns=na_column_to_drop, inplace=True)

           rows_to_drop = df.index[df.isnull().sum(axis=1) >= DataPreprocessingApp.missing_values_threshold]
           df.drop(rows_to_drop, inplace=True) 
        if df.shape[0] < 2:
            df_encoded = pd.get_dummies(df, drop_first=True, dummy_na=True)
        df_encoded = pd.get_dummies(df, drop_first=True)
        return df_encoded
    
    def data_train_test_split(self, df_encoded):
        X = df_encoded.drop(columns='Loan_Status_Y')
        y = df_encoded['Loan_Status_Y']    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # if self.is_na:
        X_train, X_test = self.missing_value_imputer(X_train, X_test)
        
        split_data_ = split_data(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
   
        return split_data_
        
    @staticmethod
    def missing_value_imputer(X_train, X_test):
        imp = SimpleImputer(strategy='mean')
        imp_train = imp.fit(X_train)
        X_train = imp_train.transform(X_train)
        X_test = imp_train.transform(X_test)
        return X_train, X_test
    
    @staticmethod
    def models_params():
        models1_ = {
            'RandomForestClassifier': RandomForestClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'LogisticRegression': LogisticRegression()
        }
    
        params1_ = { 
            'RandomForestClassifier': [
                {'n_estimators': [16, 32], "min_samples_leaf": [10]},
                {'criterion': ['gini', 'entropy'], 'n_estimators': [8, 16]}
            ],
            'AdaBoostClassifier': {'n_estimators': [16, 32]},
            'GradientBoostingClassifier': {'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0]},
            'DecisionTreeClassifier': {'max_depth': [1, 3, 5], 'min_samples_leaf': [30, 35, 40]},
            'LogisticRegression': {'solver': ['liblinear']}
        }
    
        model_params_ = model_params(
            models=models1_,
            params=params1_
        )
        return model_params_
    
    def train_models(self,split_data_):
        model_params_obj = self.models_params()
        models = model_params_obj.models
        params = model_params_obj.params
        
        helper = EstimatorSelectionHelper(models, params)

        X = split_data_.X_train
        y = split_data_.y_train

        helper.fit(X, y, scoring='f1_macro', n_jobs=-1)
        
        results = []
        best_score = 0.0
        
        for threshold in self.threshold_values:
            y_pred, _ = helper.predict_proba_with_threshold(X, threshold)
            score = f1_score(y, y_pred, average='macro')
            results.append((threshold, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        best_estimator = helper.result[0]['classifier']
        best_score = results[0][1]
        
        for result in results:
            if result[1] > best_score:
                best_score = result[1]
                best_estimator = helper.result[result[0]]['classifier']
        
        return results, best_estimator
    
    def save_model(self, best_estimator):
        self.model_path = Path(os.path.join(self.root_dir,'loan_eligibility_model.pkl'))
        joblib.dump(best_estimator, self.model_path)
    
    def predict_loan_eligibility(self):

        model = joblib.load(self.model_path)

        st.title("Loan Eligibility Prediction")
        
        st.write("If the data to be predicted in csv file please upload the file")
        # Allow user to upload a file
        file = st.file_uploader("Upload a CSV file", type=["csv"])
        if file is not None:
            file_data = pd.read_csv(file)

            processed_user_data = self.preprocessing(file_data)
            if "Loan_Status_Y" in processed_user_data.columns:
                processed_user_data.drop(columns=["Loan_Status_Y"],inplace=True)
            imp = SimpleImputer(strategy="mean")
            imputed_data= imp.fit_transform(processed_user_data)
            prediction = model.predict(imputed_data)

            prediction_df = pd.DataFrame(prediction, columns=["Loan_Status_Prediction"])
            # Concatenate the 'file_data' DataFrame with the predictions DataFrame horizontally
            predicted_df = pd.concat([file_data, prediction_df], axis=1)

             # Display result in browser   
            st.write("First Five Records of Predictions")                
            st.dataframe(predicted_df[:5])

            st.write("Prediction of the First Record in the Data")
            if prediction[0] == 1:
                st.write("Congratulations! You are eligible for the loan.")
            else:
                st.write("Sorry, you are not eligible for the loan.")

            #Saving results
            st.write("Predicted Records Stored at : ./Answer_6/user_input.csv")
            file_data.to_csv("./Answer_6/user_input.csv", index=False)  
        
        st.write("ALTERNATIVE to input the data for prediction")
        # Add input fields for user data
        Loan_ID = st.text_input("Loan ID")
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Married = st.selectbox("Married", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
        ApplicantIncome = st.number_input("Applicant Income")
        CoapplicantIncome = st.number_input("Co-applicant Income")
        LoanAmount = st.number_input("Loan Amount")
        Loan_Amount_Term = st.number_input("Loan Amount Term")
        Credit_History = st.selectbox("Credit History", ["0", "1"])
        Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])


        # Create a dictionary to store the user input
        user_data = {
            'Loan_ID': [str(Loan_ID)],
            'Gender': [str(Gender)],
            'Married':  [str(Married)],
            'Dependents': [str(Dependents)],
            'Education': [str(Education)],
            'Self_Employed': [str(Self_Employed)],
            'ApplicantIncome': [int(ApplicantIncome)],
            'CoapplicantIncome': [float(CoapplicantIncome)],
            'LoanAmount': [float(LoanAmount)],
            'Loan_Amount_Term': [float(Loan_Amount_Term)],
            'Credit_History': [int(Credit_History)],
            'Property_Area': [str(Property_Area)]
        }
   
        user_df = pd.DataFrame(user_data)
        st.dataframe(user_df)
        processed_user_data = self.preprocessing(user_df)  # Apply the same preprocessing steps used during training
        print(f"Preprocessed Data :{ processed_user_data}")
        st.dataframe(processed_user_data)

        # Display the submit button
        if st.button("Submit"):    
            
            prediction = model.predict(processed_user_data)

            # Store the user input in a file
            user_df.to_csv("user_input.csv", mode='a+', index=False)  
            
            # Display the prediction result
            if prediction[0] == 1:
                st.write("Congratulations! You are eligible for the loan.")
            else:
                st.write("Sorry, you are not eligible for the loan.")

      

    def initiate_model_training(self):
        df = self.get_data()
        df_encoded = self.preprocessing(df=df)
        split_data_ = self.data_train_test_split(df_encoded=df_encoded)
        results, best_estimator = self.train_models(split_data_)
        self.save_model(best_estimator=best_estimator)
        
    def __del__(self):
        print(f"{'>>'*20} Training complete {'<<'*20}")

# @st.cache_data
def run_app():
    data_preprocessing_app = DataPreprocessingApp()
    data_preprocessing_app.initiate_model_training()
    data_preprocessing_app.predict_loan_eligibility()

if __name__ == "__main__":
    run_app()
