import numpy as np
import pandas as pd

melbourne = pd.read_csv("/home/jhojin/CODE/trying-tensorflow/data/melb_data.csv")
# print(melb.columns)
# print(melb.info())


melb = melbourne.dropna(0)

features = ['Rooms','Bathroom','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
# 'Distance','Bedroom2','Propertycount'
X = melb[features]
print(X)
y = melb.Price
print(y,type(y))

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X,y)

# error = actual - predicted
from sklearn.metrics import mean_absolute_error
prediction = model.predict(X)
print(prediction)
mae = mean_absolute_error(y,prediction)
print(mae)

