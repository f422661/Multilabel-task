from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score
from multilabelFunc import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
       
        self.fc1 = nn.Linear(365, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 492)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class CustomizedDataset(Dataset):

    def __init__(self,x,y):

        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]



if __name__ == '__main__':

    x = np.load('./dataset1/X.npy')
    y = np.load('./dataset1/y.npy')

    kf = KFold(n_splits=3,shuffle=True)


    acc_list = list()
    precision_list = list()
    recall_list = list()
    exact_acc_list = list()


    for train_index, test_index in kf.split(x):

        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]


        train_data = CustomizedDataset(x_train,y_train)
        data_loader = DataLoader(train_data, batch_size=500,shuffle=True)


        # Build the network
        net = Net()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters())

        # params = list(net.parameters())
        # print(net)


        for epoch in range(100):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(data_loader):

                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                ## forward + backward + optimize
                # print(inputs)
                outputs = net(inputs.float())


                print("!!!!!!!")
                print(outputs.shape)



                loss = criterion(outputs,labels.float())

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                if i % 10 == 9:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

        print('Finished Training')

        ## Evaluation
        with torch.no_grad():

            x_test = torch.from_numpy(x_test)
            nn_pred = np.round(net(x_test.float()).numpy())
            # print(outputs)
            # print(outputs.numpy())

            example_accuracy = accuracyMultilabel(y_test,nn_pred)
            example_precision = precisionMultilabel(y_test,nn_pred)
            example_recall = recallMultilabel(y_test,nn_pred)
            exact_acc = accuracy_score(y_test,nn_pred)

            print("example-based accuracy:%f"%example_accuracy)
            print("example-based precision:%f"%example_precision)
            print("example-based recall:%f"%example_recall)
            print("Exact accuracy:%f"%exact_acc)

            break

        # for x_batch,y_batch in dataloader:
        #   print(x_batch.shape)
        #   print(y_batch.shape)
        #   break

