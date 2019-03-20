import tensorflow as tf  
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


## Let print no ignore
np.set_printoptions(threshold=np.nan)

X = np.load('./dataset1/X.npy')
y = np.load('./dataset1/y.npy')

print('X shape :{0}'.format(X.shape))
print('y shape :{0}'.format(y.shape))


kf = KFold(n_splits=3,shuffle=True)

for train_index, test_index in kf.split(X):

    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = MLPClassifier(hidden_layer_sizes=(300,300,300),max_iter= 1000)
    clf.fit(X_train,y_train) 
    x_pred = clf.predict(X_test)

    print(clf.score(X_test,y_test))

