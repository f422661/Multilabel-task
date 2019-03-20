import tensorflow as tf  
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix


def multiConfusion_matrix(y_true,y_pred,label_name):

    labels = label_name
    conf_mat_dict= {}

    for label_col in range(len(labels)):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]
        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
    
    return conf_mat_dict


## Let print no ignore
np.set_printoptions(threshold=np.nan)

# Training Parameters
learning_rate = 0.001
epochs = 200
batch_size = 500

# Network Parameters
num_input = 365
n_neurons = 300 # hidden layer num of features
n_outputs = 492 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [None,num_input])
Y = tf.placeholder(tf.float32, [None,n_outputs])

## feedforward network
hidden1 = tf.maximum(tf.layers.dense(X,n_neurons),0)
hidden2 = tf.maximum(tf.layers.dense(hidden1,n_neurons),0)
logits = tf.layers.dense(hidden2,n_outputs)
pred = tf.round(tf.nn.sigmoid(logits))


# Define loss and optimizer
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
loss_op = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


## Evaluate model
# correct_pred = tf.equal(pred,Y)
correct_pred = tf.equal(tf.round(tf.reshape(pred,[-1])),tf.reshape(Y,[-1]))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

x = np.load('./dataset1/X.npy')
y = np.load('./dataset1/y.npy')

## shuffle data
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

print('x shape :{0}'.format(x.shape))
print('y shape :{0}'.format(y.shape))

kf = KFold(n_splits=3,shuffle=True)

if __name__ == '__main__':


    for train_index, test_index in kf.split(x):

        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]


        with tf.Session() as sess:

            print("# of training data:",x_train.shape[0])

            sess.run(init)

            for e in range(epochs):

                loss_epoch = list()
                acc_epoch = list()

                print("epochs",e+1)

                for i in range(int(len(x_train)/batch_size)+1):

                    if i == (int(len(x_train)/batch_size)) and len(x_train)%batch_size == 0:
                        break

                    # print(str(i*batch_size)+"/"+str(len(x_train)))

                    x_batch = x_train[i*batch_size:(i+1)*batch_size]
                    y_batch = y_train[i*batch_size:(i+1)*batch_size]

                    # Run optimization op (backprop)
                    sess.run(train_op,feed_dict={X: x_batch, Y: y_batch})
                    loss_ = loss_op.eval(feed_dict={X: x_batch, Y: y_batch})
                    acc = accuracy.eval(feed_dict={X: x_batch, Y: y_batch})

                    nn_pred = pred.eval(feed_dict={X: x_batch, Y: y_batch})

                    # print("debug")
                    # print(nn_pred[0])
                    # print(y_batch[0])

                    loss_epoch.append(loss_)
                    acc_epoch.append(acc)

                    
                print("loss:",np.mean(np.array(loss_epoch)),"acc:",np.mean(np.array(acc_epoch)))
                # print("loss:",np.mean(np.array(loss_epoch)))

            ## evaluation
            nn_pred = pred.eval(feed_dict={X: x_test}) 

            acc_test = accuracy.eval(feed_dict={X: x_test,Y: y_test})


            prec_micro = precision_score(y_test,nn_pred,average='micro')
            recall_micro = recall_score(y_test,nn_pred,average='micro')
            prec_macro = precision_score(y_test,nn_pred,average='macro')
            recall_macro = recall_score(y_test,nn_pred,average='macro')

            print("accuracy:%f" %acc_test)
            print("micro precision:%f" %prec_micro)
            print("micro recall:%f" %recall_micro)
            print("macro precision:%f" %prec_macro)
            print("macro recall:%f" %recall_macro)


            # for i in range(y_test.shape[0]):  
            #     print("label:%d"%i)
            #     tn, fp, fn, tp = confusion_matrix(y_test[:,i],nn_pred[:,i]).ravel()
            #     print('tn:%d fp:%d fn:%d tp %d'%(tn,fp,fn,tp))
            #     print(precision_score(y_test[:,i],nn_pred[:,i],average='binary'))
            #     print(recall_score(y_test[:,i],nn_pred[:,i],average='binary'))


            # print(confusion_matrix(y_test[:,0],nn_pred[:,0]))
            # multi_matrix = multi_confusion_matrix(y_test,nn_pred,np.arange(n_outputs))


            # for key,value in multi_matrix.items():
            #     print(key)
            #     print(value)
            # print(multilabel_confusion_matrix(y_test,nn_pred))



            # test_acc = accuracy_multilabel(y_test,nn_pred)
            # test_prec = precision_multilabel(y_test,nn_pred)
            # test_recall = recall_multilabel(y_test,nn_pred)

            # print("accuracy: %f" %test_acc)
            # print("precision: %f" %test_prec)
            # print("recall: %f" %test_recall)





