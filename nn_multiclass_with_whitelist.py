import tensorflow as tf  
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing



## Let print no ignore
np.set_printoptions(threshold=np.nan)

# Training Parameters
learning_rate = 0.001
epochs = 5
batch_size = 500

# Network Parameters
num_input = 491
n_neurons = 300 # hidden layer num of features
# n_rnn_layers = args.l # hidden layer num of features
n_outputs = 14 # MNIST total classes (0-9 digits)

# Define the computation graph
X = tf.placeholder(tf.float32, [None,num_input])
Y = tf.placeholder(tf.float32, [None,n_outputs])

## feedforward network
hidden1 = tf.maximum(tf.layers.dense(X,n_neurons),0)
hidden2 = tf.maximum(tf.layers.dense(hidden1,n_neurons),0)
# hidden3 = tf.maximum(tf.layers.dense(hidden2,n_neurons),0)
logits = tf.layers.dense(hidden2,n_outputs)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

x = np.load('./dataset4/X1.npy')
y = np.load('./dataset4/y1.npy')


print(y)


# whiteList = [1,4,6,8,12,13,15,16,17,18,19,23,24,27]

# x_new = list()
# y_new = list()

# for i in range(len(y)):
#     if y[i] in whiteList:
#         x_new.append(x[i])
#         y_new.append(y[i])

# x_new = np.array(x_new)
# y_new = np.array(y_new)

# x = x_new
# y = y_new




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


kf = KFold(n_splits=3,shuffle=True)

acc1 = [0.0]*14
acc_total = list()

if __name__ == '__main__':


    for train_index, test_index in kf.split(x):

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

            print("test accuracy:{}".format(test_acc))

            for i in range(14):

                temp_x = list()
                temp_y = list()

                y_decode = np.argmax(y_test, axis=1)
                # print(y_decode)
                for index,item in enumerate(y_decode):
                    if i == item:
                        temp_x.append(x_test[index])
                        temp_y.append(y_test[index])

                temp_x = np.array(temp_x)
                temp_y = np.array(temp_y)
                # print(temp_x.shape)
                # print(temp_y.shape)
                # print(temp_x)
                acc_test = accuracy.eval(feed_dict={X: temp_x, Y: temp_y})
                acc1[i] = acc_test

            acc_total.append(acc1)
       
    print(acc_total)
    print(np.mean(np.array(acc_total), axis=0))