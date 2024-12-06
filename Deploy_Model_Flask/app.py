from flask import Flask, render_template, request
import numpy as np 
import joblib

app = Flask(__name__)
model = joblib.load('RandomForestModel.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ptratio = float(request.form['PTRATIO'])
        lstat = float(request.form['LSTAT'])
        rm = float(request.form['RM'])
        rad = float(request.form['RAD'])
        crim = float(request.form['CRIM'])


        data=np.array([ptratio,lstat,rm,rad,crim]).reshape(1,5)


        prediction = model.predict(data)
        return render_template('result.html', prediction= prediction )

if __name__ == '__main__':
    app.run(debug=True,port=5000)
