import numpy as np
import pandas as pd

titanic = pd.read_csv("./data/spaceship-titanic/train.csv")

# define y. prediction target
titanic.columns
y = titanic.Transported

# define X. features
feature_names = list(titanic.columns)
# feature_names.remove("Transported")
feature_names = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X = titanic[feature_names]

# describe data
print(X.info())
print(X.isna())
# print(X.describe())
print(X.head())

# make model
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)
model.fit(X,y)
