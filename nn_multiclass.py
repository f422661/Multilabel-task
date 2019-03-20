import tensorflow as tf  
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn import preprocessing


## Let print no ignore
np.set_printoptions(threshold=np.nan)

# Training Parameters
learning_rate = 0.001
epochs = 100
batch_size = 500

# Network Parameters
num_input = 368
n_neurons = 300 # hidden layer num of features
# n_rnn_layers = args.l # hidden layer num of features
n_outputs = 28 # MNIST total classes (0-9 digits)

# Define the computation graph
X = tf.placeholder(tf.float32, [None,num_input])
Y = tf.placeholder(tf.float32, [None,n_outputs])

## feedforward network
hidden1 = tf.maximum(tf.layers.dense(X,n_neurons),0)
hidden2 = tf.maximum(tf.layers.dense(hidden1,n_neurons),0)
# hidden3 = tf.maximum(tf.layers.dense(hidden2,n_neurons),0)
logits = tf.layers.dense(hidden2,n_outputs)
pred = tf.argmax(tf.nn.softmax(logits),1)


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(Y, 1))



accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

x = np.load('./dataset3/X1.npy')
y = np.load('./dataset3/y1.npy')

## shuffle data
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

## Convert y to ont hot representation.
lb = preprocessing.LabelBinarizer()
lb.fit(y)
y_onehot = lb.transform(y)

print('x shape :{0}'.format(x.shape))
print('y shape :{0}'.format(y_onehot.shape))

kf = StratifiedKFold(n_splits=3,shuffle=True)

acc1 = [0.0]*n_outputs
acc_total = list()
acc_test_list = list()

if __name__ == '__main__':


    precision_list = list()
    recall_list = list()


    for train_index, test_index in kf.split(x,y):

        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_onehot[train_index], y_onehot[test_index]

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

                    # nn_pred = pred.eval(feed_dict={X: x_batch, Y: y_batch})

                    # print("debug")
                    # print(nn_pred[0])
                    # print(y_batch[0])

                    loss_epoch.append(loss_)
                    acc_epoch.append(acc)
   
                print("loss:",np.mean(np.array(loss_epoch)),"acc:",np.mean(np.array(acc_epoch)))

            test_acc = accuracy.eval(feed_dict={X: x_test, Y: y_test})
            nn_pred = pred.eval(feed_dict={X: x_test}) 

            # print(nn_pred[0:5])

            # precision = precision_score(y_test,nn_pred)
            # recall = recall_score(y_test,nn_pred)

            

            # precision_list.append(precision)
            # recall_list.append(recall)
            acc_test_list.append(test_acc)


            print("Accuracy:{}".format(test_acc))
            # print("Precision:%f"%precision)
            # print("Recall:%f"%recall)




            # for i in range(n_outputs):

            #     temp_x = list()
            #     temp_y = list()

            #     y_decode = np.argmax(y_test, axis=1)
            #     # print(y_decode)
            #     for index,item in enumerate(y_decode):
            #         if i == item:
            #             temp_x.append(x_test[index])
            #             temp_y.append(y_test[index])

            #     temp_x = np.array(temp_x)
            #     temp_y = np.array(temp_y)


            #     print(temp_x.shape)
            #     print(temp_y.shape)
            #     # print(temp_x)


            #     acc_test = accuracy.eval(feed_dict={X: temp_x, Y: temp_y})
            #     acc1[i] = acc_test

            # acc_total.append(acc1)
       
    # print(acc_total)
    # print(np.mean(np.array(acc_total), axis=0))

    print("The mean of the test accuracy:%f"%np.mean(acc_test_list))
    # print("The mean of the test precision:%f"%np.mean(precision_list))
    # print("The mean of the test recall:%f"%np.mean(recall_list))