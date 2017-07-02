import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

test_file = 'test.csv'

# Example row:
# 04/01/2016,52.00,1675437.00

dates  = []
prices = []
volume = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			prices.append(float(row[1]))
			volume.append(int(row[2]))
	return
def predict_prices(dates, prices, x):
	dates = np.reshape(dates,len(dates),1)

	svr_lin  = SVR(kernel = 'linear', C=1e3)
	svr_poly = SVR(kernel = 'poly',   C=1e3, degree = 2)
	svr_rbf  = SVR(kernel = 'rbf',    C=1e3, gamma  = 0.1)
	svr_lin.fit(dates,prices)
	svr_poly.fit(dates,prices)
	svr_rbf.fit(dates,prices)

	plt.scatter(dates, prices, color='black', label='Data')
	plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
	plt.plot(dates, svr_lin.predict(dates, color='green', label='Linear Model'))
	plt.plot(dates, svr_poly.predict(dates, color='blue', label='Polynomial Model'))
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('SVM')
	plt.legend()
	plt.show	

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data(test_file)
predictedPrice = predict_prices(dates, prices, 29)

print(dates)
print(prices)
print(volume)