import pandas as pd
import pickle

data = pd.read_csv('taxi.csv')
print(data.head())

data_x = data.iloc[:,0:-1].values
data_y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

print("Training Score:", regressor.score(X_train,y_train))
print("Testing Scores:", regressor.score(X_test, y_test))

pickle.dump(regressor, open('model.pkl', 'wb'))
