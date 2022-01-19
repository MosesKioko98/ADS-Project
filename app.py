import pandas as pd
df = pd.read_csv("diabetes.csv")
#df.info()
# checking for missing values
df.isnull().sum()
# creating a copy of the dataset
df1=df.copy()
# Descriptive statistics
df1.describe()
import numpy as np
# convert the zeros to null values
df1[['Glucose','BloodPressure','SkinThickness','Insulin', 'BMI']] = df1[['Glucose','BloodPressure','SkinThickness','Insulin', 'BMI']].replace(0,np.NaN)
# checking for missing values
df1.isnull().sum()
df1.to_csv("df1.csv",index=False)

#import matplotlib.pyplot as plt
# creating an histogram for each numeric feature
numeric_features=df1.drop("Outcome",axis=1)
#for col in numeric_features:
    #fig=plt.figure(figsize=(9,6))
    #ax=fig.gca()
    #feature=df1[col]
    #feature.hist(ax=ax)
    #ax.axvline(feature.mean(),color = 'magenta',linestyle='dashed',linewidth=2)
    #ax.axvline(feature.median(),color = 'cyan',linestyle='dashed',linewidth=2)
    #ax.set_title(col)
#plt.show()
# Data Preprocessing
# 1. Imputation
df1["Glucose"].fillna((df1["Glucose"].mean()),inplace=True)
df1["BloodPressure"].fillna((df1["BloodPressure"].mean()),inplace=True)
df1["SkinThickness"].fillna((df1["SkinThickness"].mean()),inplace=True)
df1["Insulin"].fillna((df1["Insulin"].median()),inplace=True)
df1["BMI"].fillna((df1["BMI"].mean()),inplace=True)
# confirming if the dataset does not have missing values
#df1.isnull().sum()
# 2. Categorical encoding
#df1["Outcome"].value_counts()
#df1.info()
# converting the target variables to category
df1["Outcome"] = df1["Outcome"].astype("category").cat.as_ordered()
#df1.info()
# 3. Checking for outliers
#for col in numeric_features:
    #fig=plt.figure(figsize=(9,6))
    #ax=fig.gca()
    #df1.boxplot(col,ax=ax)
#plt.show()
# 4. Feature selection
#cor=df1.corr()
#cor
#import seaborn as sns
#plt.figure(figsize=(12,10))
#sns.heatmap(cor,annot=True,cmap=plt.cm.Reds)
#plt.show()
# Separate the features from labels
features=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
label=["Outcome"]
x,y = df1[features].values,df1[label].values
# 5. Split the data into training and validation set
from sklearn.model_selection import train_test_split
# split data 80%-20% into training and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=0)
print('Training cases:%d\nTest cases:%d'%(x_train.shape[0],x_test.shape[0]))
# 6. Train the model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
# Define preprocessing for numeric columns(normalize so that they are on the same scale)
numeric_features = [0,1,2,4,5,6,7]
numeric_transformer = Pipeline(steps=[('scaler',MinMaxScaler())])
preprocessor=ColumnTransformer(transformers=[('num',numeric_transformer,numeric_features)])
# create a training pipeline
reg=0.01
pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                          ('logregressor',LogisticRegression(C=1/reg,solver="liblinear"))])
# fit the pipeline to train a logistic regression on the training set
model=pipeline.fit(x_train,y_train)
# Checking for perfomance of the model on both the trainig and validation set
from sklearn.metrics import accuracy_score
predictions=model.predict(x_test)
print('Accuracy',accuracy_score(y_test,predictions))
x_train_predictions=model.predict(x_train)
print('Accuracy',accuracy_score(y_train,x_train_predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
# We can retrive precision and recall values on their 
from sklearn.metrics import precision_score,recall_score
print("Overall Precision:",precision_score(y_test,predictions))
print("Overall Recall:",recall_score(y_test,predictions))
from sklearn.metrics import confusion_matrix
# print the confusion matrix
#cm=confusion_matrix(y_test,predictions)
#fig=sns.heatmap(pd.DataFrame(cm),annot=True,cmap=plt.cm.Blues)
#plt.title("confusion_matrix")
#plt.ylabel("Actual Label")
#plt.xlabel("Predicted Label")
from sklearn.metrics import roc_auc_score
y_score=model.predict_proba(x_test)
auc = roc_auc_score(y_test,y_score[:,1])
print('AUC:'+ str(auc))
import joblib 
import pickle
import streamlit as st
# loading the trained model
filename = './diabetic_model.pkl'
joblib.dump(model,filename)
classifier=joblib.load(filename)
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):  
 
    # Pre-processing user input    
    Pregnancies=Pregnancies
    
    Glucose=Glucose
    
    BloodPressure=BloodPressure
    
    SkinThickness=SkinThickness
    
    Insulin=Insulin
    
    BMI=BMI
    
    DiabetesPedigreeFunction=DiabetesPedigreeFunction
    
    Age=Age
 
    
    # Making predictions 
    prediction = classifier.predict( 
        [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
     
    if prediction == 0:
        pred = 'Non Diabetic'
    else:
        pred = 'Diabetic'
    return pred
      
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Diabetic Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Pregnancies= st.number_input("Total Pregnancies")
    Glucose= st.number_input("Glucose Level")
    BloodPressure = st.number_input("BloodPressure")
    SkinThickness = st.number_input("SkinThickness")
    Insulin = st.number_input("Insulin")
    BMI = st.number_input("BMI")
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction")
    Age = st.number_input("Age")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age) 
        st.success('Your status is {}'.format(result))
        
if __name__=='__main__':
    main()
