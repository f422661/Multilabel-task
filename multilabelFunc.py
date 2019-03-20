import numpy as np
import copy
from sklearn.metrics import confusion_matrix


## Example-based accuracy, precision, recall using in multilabel classification
def accuracyMultilabel(y_true,y_pred):

    matrices_and = np.logical_and(y_true,y_pred).astype(float)
    matrices_or = np.logical_or(y_true,y_pred).astype(float)

    N = y_true.shape[0]
    count = 0
    for i in range(N):
        count += np.sum(matrices_and[i])/np.sum(matrices_or[i])
    return count/N

def precisionMultilabel(y_true,y_pred):

    matrices_and = np.logical_and(y_true,y_pred).astype(float)
    matrices_or = np.logical_or(y_true,y_pred).astype(float)

    N = y_true.shape[0]
    count = 0
    ignore_count = 0
    for i in range(N):

        ## The number of model prediciton could be zero.
        if np.sum(y_pred[i]) == 0:
            print("sample %d is ignored"%i)
            ignore_count+=1

            continue
            
        count += np.sum(matrices_and[i])/np.sum(y_pred[i])


    return count/(N-ignore_count)

def recallMultilabel(y_true,y_pred):

    matrices_and = np.logical_and(y_true,y_pred).astype(float)
    matrices_or = np.logical_or(y_true,y_pred).astype(float)

    N = y_true.shape[0]
    count = 0
    for i in range(N):
        count += np.sum(matrices_and[i])/np.sum(y_true[i])
    return count/N

def multiConfusionMatrix(y_true,y_pred,label_name):

    labels = label_name
    conf_mat_dict= {}

    for label_col in range(len(labels)):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]
        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
    
    return conf_mat_dict

def precisionRecallCurve(y_true,y_pred):

    thresh = np.arange(0.1,1,0.1)
    # thresh = [0.4]
    precisoin = list()
    recall = list()

    for t in thresh:

        print("t:%f"%t)
        temp = copy.deepcopy(y_pred)
        temp[temp>=t] = 1
        temp[temp<t] = 0

        precisoin.append(precisionMultilabel(y_true,temp))
        recall.append(recallMultilabel(y_true,temp))


    return precisoin,recall









