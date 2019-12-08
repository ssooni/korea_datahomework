# load data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# train
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


use_cuda = torch.cuda.is_available()
summary = SummaryWriter()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 300))  # 6@24*24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(2, 1))
        self.fc1 = nn.Linear(7952, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 72)

        self.conv_module = nn.Sequential(
            self.conv1,
            self.conv2
        )

        self.fc_module = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3
        )

    def forward(self, x):
        out = self.conv_module(x)
        # make linear
        dim = 1
        # print(out.size())
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)
        # print(out.size())
        out = self.fc_module(out)
        return F.softmax(out, dim=1)


class NewsDataSet(Dataset):
    def __init__(self, npy_file_name, size=500):
        super(NewsDataSet, self).__init__()
        self.docData = np.load(npy_file_name, allow_pickle=True)
        self.x, self.y = self.preprocessing(size)
        self.length = len(self.docData)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.length

    def preprocessing(self, size):
        # Zevro-Padding : CNN은 고정길이를 입력으로 하기 떄문에
        count = 1
        x = list()
        y = list()
        lable_dict = dict()
        index = 0
        for tag, docVec in self.docData:
            if docVec.shape[0] == 0:
                continue

            elif docVec.shape[0] < size:
                padSize = size - docVec.shape[0]
                padding = np.zeros((padSize, 300))
                docVec = np.concatenate((docVec, padding), axis=0)
            else:
                docVec = docVec[0:size]

            x.append(docVec)
            label = tag.split(">")[0]
            if label not in lable_dict:
                lable_dict[label] = index
                index += 1

            y.append(lable_dict[label])

            if count % 100 == 0:
                print(count / len(self.docData) * 100)

            count += 1
        print(index)
        return x, y



def train():
    cnn = CNN()
    newsData = NewsDataSet("./article.npy")

    batch_size = 1
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(newsData)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trn_loader = torch.utils.data.DataLoader(newsData, sampler=train_sampler, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(newsData, sampler=valid_sampler, batch_size=batch_size)

    # loss
    criterion = nn.NLLLoss()

    # backpropagation method
    learning_rate = 1e-4
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    # hyper-parameters
    num_epochs = 10
    num_batches = len(trn_loader)

    print(num_batches)

    trn_loss_list = []
    val_loss_list = []
    for epoch in range(num_epochs):
        trn_loss = 0.0

        for i, data in enumerate(trn_loader):
            x, label = data

            x = x.unsqueeze(0).double()
            label = label.long()

            # grad init
            optimizer.zero_grad()

            # forward propagation
            cnn = cnn.double()
            model_output = cnn(x)

            # calculate loss
            loss = criterion(model_output,  label)

            # back propagation
            loss.backward()

            # weight update
            optimizer.step()

            # trn_loss summary
            trn_loss += loss.item()
            # del (memory issue)
            del loss
            del model_output

            # 학습과정 출력
            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    val_loss = 0.0
                    for j, val in enumerate(val_loader):
                        val_x, val_label = val
                        val_x = val_x.unsqueeze(0).double()
                        val_label = val_label.long()

                        val_output = cnn(val_x)

                        v_loss = criterion(val_output, val_label)
                        val_loss += v_loss

                    print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format( epoch + 1, num_epochs, i + 1, num_batches, trn_loss / 100, val_loss / len(val_loader)))

                trn_loss_list.append(trn_loss / 100)
                val_loss_list.append(val_loss / len(val_loader))
                trn_loss = 0.0
    torch.save(cnn.state_dict(), "./cnn.model")


def val():
    checkpoint = torch.load("./cnn.model")
    cnn = CNN()
    cnn.load_state_dict(checkpoint)
    cnn.eval()
    print(cnn.eval())
    newsData = NewsDataSet("./article.npy")

    count = 0
    for i in range(newsData.length):
        x, y = newsData[i][0], newsData[i][1]
        val_x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
        val_output = cnn(val_x)
        values, indices = val_output[0].max(0)
        print(y, indices)
        if y == indices:
            count += 1

    print(count / newsData.length)




if __name__ == '__main__':
    val()