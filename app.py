# importing necessary libraries and functions
import numpy as np
# import flask
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
# Use pickle to load in the pre-trained model.

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    if request.method == 'POST':
        
        init_features = [float(x) for x in request.form.values()]
        final_features = np.array(pd.Series(init_features, index= None)).reshape(1, -1)
        print(init_features)
        print(final_features)
        # print(final_features.shape)
        prediction = model.predict(final_features) # making prediction
        print(prediction)
            # return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction)) # rendering the predicted result
        if prediction == 0:
            return render_template('index.html',prediction_text='Your next purchase day is in 50 days and above')
        elif prediction == 1:
            return render_template('index.html',prediction_text='Your next purchase day is between 20-50 days')
        else:
            return render_template('index.html', prediction_text='Your next purchase day is in less than 20 days')



if __name__ == "__main__":
    app.run(debug=True)



