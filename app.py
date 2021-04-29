from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('healthcare_premium_standard_scalar.pkl')
scaler = joblib.load('standard_scalar.pkl')

# you are routing your web page to go through this app so that we can connect it with python

@app.route('/prediction', methods = ['POST'])
def prediction():
    age = request.form['age']
    bmi = request.form['bmi']
    # converting string form values to floating point
    age, bmi = float(age), float(bmi)
    scaled_values = scaler.transform([[age, bmi]])
    result = model.predict(scaled_values)
    string = 'The Healtcare Premium Charges are: ' + str(result[0])
    return render_template('index.html', prediction_text = string)

# running the app
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, threaded = False)