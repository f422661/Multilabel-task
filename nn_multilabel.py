import tensorflow as tf  
import numpy as np
from multilabelFunc import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

## Let print no ignore
np.set_printoptions(threshold=np.nan)

# Training Parameters
learning_rate = 0.001
epochs = 1
batch_size = 500

# Network Parameters
num_input = 366
n_neurons = 300 # hidden layer num of features
n_outputs = 495 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [None,num_input])
Y = tf.placeholder(tf.float32, [None,n_outputs])

## feedforward network
hidden1 = tf.maximum(tf.layers.dense(X,n_neurons),0)
hidden2 = tf.maximum(tf.layers.dense(hidden1,n_neurons),0)
logits = tf.layers.dense(hidden2,n_outputs)
pred = tf.nn.sigmoid(logits)


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

x = np.load('./dataset6/tX.npy')
y = np.load('./dataset6/ty.npy')

## shuffle data
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

print('x shape :{0}'.format(x.shape))
print('y shape :{0}'.format(y.shape))

kf = KFold(n_splits=3,shuffle=True)

if __name__ == '__main__':


    acc_list = list()
    precision_list = list()
    recall_list = list()
    exact_acc_list = list()
    p_list = list()
    r_list = list()

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
            aaa = hidden2.eval(feed_dict={X: x_test}) 
            print("==================")
            print(type(aaa))



            p , r = precisionRecallCurve(y_test,nn_pred)

            p_list.append(p)
            r_list.append(r)

            acc_test = accuracy.eval(feed_dict={X: x_test,Y: y_test})

            example_accuracy = accuracyMultilabel(y_test,nn_pred)
            # example_precision = precisionMultilabel(y_test,nn_pred)
            # example_recall = recallMultilabel(y_test,nn_pred)

            # print(example_precision)
            # print(example_recall)


            nn_pred = np.round(nn_pred)
            example_accuracy = accuracyMultilabel(y_test,nn_pred)
            exact_acc = accuracy_score(y_test,nn_pred)
            acc_list.append(example_accuracy)
            # precision_list.append(example_precision)
            # recall_list.append(example_recall)
            exact_acc_list.append(exact_acc)


            print("example-based accuracy:%f"%example_accuracy)
            # print("example-based precision:%f"%example_precision)
            # print("example-based recall:%f"%example_recall)
            print("Exact accuracy:%f"%exact_acc)



    print("The mean of example-based accuracy:%f"%np.mean(acc_list))
    # print("The mean of example-based precision:%f"%np.mean(precision_list))
    # print("The mean of example-based recall:%f"%np.mean(recall_list))
    print("The mean of exact accuracy:%f"%np.mean(exact_acc_list))


    print("p:{}".format(np.mean(p_list,axis=0)))

    print("r:{}".format(np.mean(r_list,axis=0)))


            # prec_micro = precision_score(y_test,nn_pred,average='micro')
            # recall_micro = recall_score(y_test,nn_pred,average='micro')
            # prec_macro = precision_score(y_test,nn_pred,average='macro')
            # recall_macro = recall_score(y_test,nn_pred,average='macro')

            # print("accuracy:%f" %acc_test)
            # print("micro precision:%f" %prec_micro)
            # print("micro recall:%f" %recall_micro)
            # print("macro precision:%f" %prec_macro)
            # print("macro recall:%f" %recall_macro)
            








