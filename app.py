import os
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from random import randrange


def price_calculator(quantity):
    price = int(quantity)  * 5
    
    return price

def price_calculator_ml_model(quantity, model_path):
    data = pd.read_csv('hour.csv')
    data = data.drop(['dteday', 'instant', 'casual', 'registered', 'cnt'], axis = 1)
    now = datetime.now()
    time = datetime.now().strftime("%H:%M:%S")
    time_hr = datetime.now().strftime('%H')
    data_by_date = data[(data['mnth'] == int(now.month)) & (data['hr'] == int(time_hr))]
    data_used = data_by_date.iloc[[randrange(0, len(data_by_date))]]
    weather_list = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain']
    weather = weather_list[data_used['weathersit'].iloc[0]]

    model = pickle.load(open(model_path, 'rb'))
    
    demand_amt = abs(int(np.round(model.predict(data_used))))

    if demand_amt > 100:
        price = price_calculator(quantity) * 2
        demand = 'Very High'
    elif demand_amt > 50:
        price = price_calculator(quantity) * 1.5
        demand = 'High'
    else:
        price = price_calculator(quantity)
        demand = 'Low'

    return price, demand

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def home():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        location = request.form['area']
        quantity = request.form['bike_qtn']

        model_path = 'model.pkl'
        price, demand = price_calculator_ml_model(quantity, model_path)

        return render_template('confirmation.html', name = name, quantity = quantity, location = location, phone = phone, price = str(price), demand = demand)

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()