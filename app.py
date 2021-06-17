# Required Libraries
from flask import Flask, render_template, request, make_response
import jsonify
import requests
import json
from requests.sessions import Request
import joblib
import numpy as np


# Importing the model
model = joblib.load('xgb_modelfinal.joblib')


app = Flask(__name__)


# Templates
# Home page
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



@app.route("/to_model", methods=['POST'])
def to_model():

    req = request.get_json()
    array = req['val_array']
    
    input_array = np.array([array])
    prediction = model.predict(input_array)

    output=prediction[0]

    outs = round(output)

    x = {"output": outs}
    y = json.dumps(x)

    return y



if __name__=="__main__":
    app.run(debug=True)

