import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("IRIS Specie Prediction")

#loading the data
@st.cache_data
def loaddata():
    iris = load_iris()
    df= pd.DataFrame(iris.data,columns=iris.feature_names)
    df['target']=iris.target
    return df , iris.target_names

df,target_names=loaddata()

#Training the Model
X_train,X_test,y_train,y_test=train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2)
model=RandomForestClassifier()
model.fit(X_train,y_train)

#slider for chossing Input
st.sidebar.title("Input Features")
sepal_length=st.sidebar.slider("sepal length",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sepal_width=st.sidebar.slider("sepal width",float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
petal_length=st.sidebar.slider("petal length",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_width=st.sidebar.slider("petal width",float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))

# prediction
specie=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
predicting_specie=target_names[specie]

st.write("PREDICTION")
st.write(f"Predicting Specie is {predicting_specie}")