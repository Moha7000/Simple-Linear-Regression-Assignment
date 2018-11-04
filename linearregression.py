import matplotlib.pyplot as mtp
import numpy as np
import pandas as pd
import statsmodels.api as sm


#read from dataset file
data = pd.read_csv('boston.csv')

x = np.array(data['crim'])
y = np.array(data['nox'])

n = np.size(x)

# 80% of data in training
x_training = np.array(x[0: int(0.8 * n) + 1])
y_training = np.array(y[0: int(0.8 * n) + 1])
# 20% of data in testing
x_test = np.array(x[int(-0.2 * n):])
y_test = np.array(y[int(-0.2 * n):])


#print (x_training)
#print (x_test)


#finding the coffecients  y = mx + c    using least square error technique
def find_coef(x, y):
    n = np.size(x)
    m = (np.mean(x) * np.mean(y) - np.mean(x * y)) / (np.mean(x) * np.mean(x)  - np.mean(x ** 2))
    c = np.mean(y) - m * np.mean(x)
    return (m, c)

m,c = find_coef(x_training, y_training)

reg_line = [(m * i) + c for i in x_training]

y_predict = np.array([(m * i) + c for i in x_test])
	
#error average
mean_error = np.mean(sum((y_predict - y_test) ** 2))

print(mean_error)

#plotting results
#drawing the training set and the model
mtp.plot(x, y, 'g.', label = 'data') # draw data as points
mtp.plot(x_training, reg_line, 'r-', label = 'Linear reg') #draw our line
mtp.plot(x_test, y_test, 'bx' , label='testing 20%')

#drwing features
mtp.title('Simple Linear reg')
mtp.xlabel('parameter 1')
mtp.ylabel('parameter 2')
mtp.legend(loc = 'best')
mtp.grid(True)
mtp.show()
