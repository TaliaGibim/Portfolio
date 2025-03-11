from flask import Flask, render_template, request, Response, send_from_directory, jsonify, redirect,url_for, make_response
import numpy as np
import pickle
from static.python.SuportFunctions import max_pooling
from static.python.MyFirstCNN import MLP
import os

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/NeuralNetwork', methods=['POST','GET'])
def redirect_nn():
  if request.method == 'POST':
    image = request.json['data']   #Transforming from list to a numpy array
    image = np.array(image, dtype=int)   # Reshape the flat image to its original dimensions (224x224)
    image = image.reshape(224, 224)  # Resize the image to 28x28 by polling 
    image = max_pooling(image,2)
    image = max_pooling(image,2)
    image = max_pooling(image,2)  # Flatten the resized image to 784 pixels
    image = image.flatten()
    pickle.dump(image, open('image.pkl','wb'))  # open a file, where you stored the pickled data

    # Load the trained model
    model = pickle.load(open(os.path.join('static','model','model.pkl'), 'rb'))
    model.predict(image)
    prediction = int(model.prediction)
    accuracy = round(float(model.accuracy)*100,1)
    print(prediction, accuracy)
    return jsonify({'prediction': prediction, 'accuracy': accuracy})
  return render_template('myfirstnn.html', prediction = None)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port = 5555,debug = True)