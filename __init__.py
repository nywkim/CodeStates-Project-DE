from flask import Flask, render_template, request
import joblib
import numpy as np

np.set_printoptions(precision=4)
model = joblib.load("./model/model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('start.html')

@app.route('/predict', methods=["GET","POST"])
def index():
    d1 = request.form['a']
    d2 = request.form['b']
    d3 = request.form['c']
    
    arr = np.array([[d2, d3]])
    pred = model.predict(arr)
    return render_template('after.html', name=d1 ,data=pred)
"""
@app.route('/predict/<name>')
def predict(name):
    prediction = model.predict([['key2','key3']])
    return f"{name}의 Average Rating은 {str(prediction)}입니다."
"""

if __name__ == '__main__':
    app.run(debug=True)