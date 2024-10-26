from flask import Flask, render_template, request
import numpy as np
import pickle

# loading model
model = pickle.load(open('model.pkl', 'rb'))

# flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Getting the features from the form
    features = request.form['feature']
    features = features.split(',')

    # Converting the features to numpy array
    np_features = np.asarray(features, dtype=np.float32)

    # Making the prediction
    pred = model.predict(np_features.reshape(1, -1))
    
    # Generating the message based on the prediction
    message = ['Cancerous' if pred[0] == 1 else 'Not Cancerous']
    
    # Rendering the template with the message
    return render_template('index.html', message=message[0])

if __name__ == '__main__':
    app.run(debug=True)


