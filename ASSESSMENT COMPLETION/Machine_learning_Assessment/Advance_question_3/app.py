import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns           
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import streamlit as st

df = pd.read_csv("advertising.csv")

X = df["TV"]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2,random_state=42)

X_train_sm = sm.add_constant(X_train) #include a constant term or an intercept. 
lr = sm.OLS(y_train,X_train_sm).fit()


y_train_pred = lr.predict(X_train_sm)


X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)
lr.save("model.pkl")
error = np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)

def main():
    st.title("Adgency Profit Prediction")

    input=st.number_input("TV", value=30)
    model = lr.load("model.pkl")
    
    df = pd.DataFrame(input, columns=["TV"])
    
    sm.add_constant(df)
    
    predicted_sales = model.predict(df)

    st.write(f"predicted sales ;{predicted_sales}")

    

