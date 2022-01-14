 
import pickle
import streamlit as st
 
# loading the trained model
ourfile= open('diabetic_model.pkl', 'rb') 
classifier = pickle.load(ourfile)
 
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
