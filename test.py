import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

model = pickle.load(open('model.pkl', 'rb'))
data = pd.read_csv('data.csv')
truth = data[['cnt']]
data = data.drop(['dteday', 'instant', 'casual', 'registered', 'cnt'], axis = 1)

pred = model.predict(data)
r2 = r2_score(truth, pred)

with open('ml_metric.txt') as f:
    prev_metric = float(f.read())

if r2 > prev_metric:
    print('deploy')
    with open('ml_metric.txt', 'w') as f:
        f.write(str(r2))

else:
    print('reject')