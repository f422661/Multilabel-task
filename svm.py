import tensorflow as tf  
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC

## Let print no ignore
np.set_printoptions(threshold=np.nan)

x = np.load('./dataset3/X1.npy')
y = np.load('./dataset3/y1.npy')

whiteList = [1,4,5,6,7,8,12,13,15,16,18,19,23,27]

x_new = list()
y_new = list()

for i in range(len(y)):
    if y[i] in whiteList:
        x_new.append(x[i])
        y_new.append(y[i])

x_new = np.array(x_new)
y_new = np.array(y_new)

x = x_new
y = y_new

## shuffle data
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

## Convert y to ont hot representation.
lb = preprocessing.LabelBinarizer()
lb.fit(y)
y_onehot = lb.transform(y)

y_index = np.argmax(y_onehot,axis = 1)

print('x shape :{0}'.format(x.shape))
print('y shape :{0}'.format(y_onehot.shape))


# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


acc1 = [0.0]*14
acc_total = list()

if __name__ == '__main__':

    # # print("TRAIN:", train_index, "TEST:", test_index)
    # x_train, x_test = x[train_index], x[test_index]
    # y_train, y_test = y_index[train_index], y_index[test_index]

    clf = GridSearchCV(SVC(), tuned_parameters, cv=3)
    clf.fit(x,y) 
    # print(clf.score(x_test,y_test))

    print(clf.best_params_)
    # clf = GridSearchCV(SVC(), tuned_parameters, cv=5,scoring='%s_macro' % score)










