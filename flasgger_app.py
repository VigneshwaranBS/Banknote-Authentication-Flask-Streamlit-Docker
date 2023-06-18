from flask import Flask, request
import numpy as np
import pandas as pd
import pickle

# flasgger

from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('Classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def hello():
    return "hello"


@app.route("/predict")
def predict_note_authentaion():
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

    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The prediction value is " + str(prediction)

# /apidocs
@app.route('/predict_file', methods=["POST"])
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


if __name__ == '__main__':
    app.run() 
