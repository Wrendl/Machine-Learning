import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from keras.datasets import mnist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

def main(bch_sz, lr, kr_sz_conv, kr_sz_pool, iters, epochs):

    print('Загрузка датасета...')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    class MyDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = data
            self.targets = torch.LongTensor(targets)
            self.transform = transform

        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]

            if self.transform:
                x = Image.fromarray(self.data[index].astype(np.uint8))
                x = self.transform(x)

            return x, y

        def __len__(self):
            return len(self.data)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MyDataset(x_train, y_train, transform=transform)
    trainloader = DataLoader(dataset, batch_size=bch_sz)

    testset = MyDataset(x_test, y_test, transform=transform)
    testloader = DataLoader(dataset, batch_size=bch_sz)

    class SimpleConvNet(nn.Module):
        def __init__(self):
            super(SimpleConvNet, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=kr_sz_conv)
            self.pool = nn.MaxPool2d(kernel_size=kr_sz_pool, stride=2)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kr_sz_conv)
            self.fc1 = nn.Linear(4*4*16, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            # print(x.shape)
            x = x.view(-1, 4*4*16)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = SimpleConvNet()

    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = lr
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    losses = []

    print('Обучение сети...')
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, batch in enumerate(tqdm(trainloader)):
            X_batch, y_batch = batch
            optimizer.zero_grad()
            y_pred = net(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/1000))
                losses.append(running_loss)
                running_loss = 0.0
            if i == iters:
                break
    print('Обучение завершено...')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    classes = list([i for i in range(10)])

    errs_num = 0
    total_num = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            y_pred = net(images)
            _, predicted = torch.max(y_pred, 1)
            c = (predicted == labels)
            for i in range(bch_sz):
                total_num += 1
                if c[i].item() != True:
                    errs_num += 1
                    # print(f'Error {errs_num}: Predicted - {predicted[i].item()} | Correct - {labels[i].item()}')
            for i in range(bch_sz):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            if total_num == iters:
                break

    print(f'There are {errs_num}/{total_num} errors.')

    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (classes[i], 100*class_correct[i] / class_total[i]))
    #
    # print([x/1000 for x in losses])

    return losses

def get_best_batch():
    batches = [5, 10, 30, 50]
    errors = []
    for b in batches:
        errors.append(main(bch_sz=b, lr=1e-4, kr_sz_conv=5, kr_sz_pool=2, iters = 6000, epochs=2))
    print(errors)
    min_error = min(errors)
    print(batches[errors.index(min_error)])

    plt.plot(batches, errors)
    plt.show()

def get_best_learning_rate():
    lrs = [4e-4, 8e-4, 1e-3, 4e-3]
    errors = []
    for l in lrs:
        errors.append(main(bch_sz=10, lr=l, kr_sz_conv=5, kr_sz_pool=2, iters=3000, epochs=2))
    print(errors)
    min_error = min(errors)
    print(lrs[errors.index(min_error)])

    plt.plot(lrs, errors)
    plt.show()

def get_best_kernel_size():
    kr_szs = [4, 5]
    errors = []
    for k in kr_szs:
        errors.append(main(bch_sz=10, lr=0.001, kr_sz_conv=k, kr_sz_pool=2, iters=3000, epochs=2))
    print(errors)
    min_error = min(errors)
    print(kr_szs[errors.index(min_error)])

    plt.plot(kr_szs, errors)
    plt.show()

def get_best_epochs_num():
    losses = main(bch_sz=10, lr=0.001, kr_sz_conv=5, kr_sz_pool=2, iters=3000, epochs=10)
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()

if __name__ == '__main__':
    # get_best_batch()
    # get_best_learning_rate()
    # get_best_kernel_size()
    get_best_epochs_num()