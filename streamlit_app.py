from flask import Flask, request
import numpy as np
import pandas as pd
import pickle

import streamlit as  st 

from PIL import Image

pickle_in = open('Classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

# @app.route('/')
def hello():
    return "hello"


# @app.route("/predict")
def predict_note_authentaion(variance,skewness,curtosis,entropy):
    """
    Auth bank note
    using docstring for specifictions

    ---
    parameters:
     - name: variance
       in: query
       required: true
       type: number
     - name: skewness
       in: query
       required: true
       type: number
     - name: curtosis
       in: query
       required: true
       type: number
     - name: entropy
       in: query
       required: true
       type: number
    responses:
       200:
           description: The output values

    """ 

   

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print (prediction)
    return prediction


# @app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """
    docstring specs 
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
             description: The output values

    """

    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = classifier.predict(df_test)

    return str(list(prediction))

# 


def main():
    st.title("Bank Authentication")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authentication ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.text_input("Variance","Enter Here")
    skewness = st.text_input("skewness","Enter Here")
    curtosis = st.text_input("curtosis","Enter Here")
    entropy = st.text_input("entropy","Enter Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentaion(variance,skewness,curtosis,entropy)
    st.success('The output is {}'.format(result))
    



if __name__ == '__main__':
    main()
