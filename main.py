from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")
model = pickle.load(open("RidgeModel.pickle", 'rb'))


@app.route("/")
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bathroom = request.form.get('bath')
    sqft = request.form.get('sq ft')

    # print(location, bhk, bathroom, sqft)
    user_data = pd.DataFrame([[location, sqft, bathroom, bhk]], columns=['location', 'total_sqft', 'bath', 'BHK'])
    prediction = round(model.predict(user_data)[0] * 1e5, 2)


    return str(prediction)
