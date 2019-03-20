import tensorflow as tf  
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


## Let print no ignore
np.set_printoptions(threshold=np.nan)


x = np.load('./dataset3/X1.npy')
y = np.load('./dataset3/y1.npy')


print(y)





## shuffle data
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

## Convert y to ont hot representation.
lb = preprocessing.LabelBinarizer()
lb.fit(y)
print(lb.classes_)
y_onehot = lb.transform(y)

l = [0]*28

for ele in y:
    l[ele]+=1


print(l)

