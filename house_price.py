import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Simple dataset
data = {
    "area": [1000, 1500, 2000, 1200, 1800],
    "bedrooms": [2, 3, 4, 2, 3],
    "bathrooms": [2, 2, 3, 1, 2],
    "floors": [1, 2, 2, 1, 2],
    "price": [5000000, 7500000, 10000000, 6000000, 8500000]
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'bathrooms', 'floors']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model Score:", model.score(X_test, y_test))

# prediction
print("Predicted price:", model.predict([[1700, 3, 2, 2]]))
