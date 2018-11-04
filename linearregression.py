import matplotlib 
import numpy as np
import pandas as pd
import statsmodels.api as sm


#read from dataset file
data = pd.read_csv('Boston.csv')

crim = np.array(data['crim'])
rm = np.array(data['rm'])

n = np.size(crim)

# 80% of data in training
crim_training = np.array(crim[0: int(0.8 * n) + 1])
rm_training = np.array(rm[0: int(0.8 * n) + 1])
# 20% of data in testing
crim_test = np.array(crim[int(-0.2 * n):])
rm_test = np.array(rm[int(-0.2 * n):])


#finding the coffecients  y = m x + c
def find_coef(x, y):
    n = np.size(x)
    m = (np.mean(x) * np.mean(y) - np.mean(x * y)) / (np.mean(x) * np.mean(x)  - np.mean(x ** 2))
    c = np.mean(y) - m * np.mean(x)
    return (m, c)

m,c = find_coef(crim_training, rm_training)



	
