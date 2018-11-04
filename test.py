import pandas as pd
import numpy as np
import matplotlib as mat
import statsmodels.api as sm


#read from dataset file
data = pd.read_csv('boston.csv')

crim = np.array(data['crim'])
rm = np.array(data['rm'])

n = np.size(crim)

# 80% of data in training
crim_training = np.array(crim[0: int(0.8 * n) + 1])
rm_training = np.array(rm[0: int(0.8 * n) + 1])
# 20% of data in testing
crim_test = np.array(crim[int(-0.2 * n):])
rm_test = np.array(rm[int(-0.2 * n):])


#print (crim_training)
print (crim_test)

