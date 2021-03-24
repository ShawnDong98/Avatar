import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

import numpy as np
import os

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training


from networks.classifier import Classifier

from tqdm import tqdm


train_dir = './datasets/train'
val_dir = './datasets/val'

batch_size = 32
epochs = 5
workers = 0 if os.name == 'nt' else 8


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

train_dataset = datasets.ImageFolder(train_dir, transform=trans)
val_dataset = datasets.ImageFolder(val_dir, transform=trans)


train_loader = DataLoader(
    train_dataset,
    num_workers=workers,
    batch_size=batch_size,
    shuffle = True,
)
val_loader = DataLoader(
    val_dataset,
    num_workers=workers,
    batch_size=batch_size,
    drop_last=True
)

resnet = InceptionResnetV1(
    pretrained='vggface2',
).to(device)
resnet.eval()

classifier = Classifier().to(device)
classifier.train()


optimizer = optim.Adam(classifier.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

loss_fn = torch.nn.BCELoss()


print('\n\nInitial')
print('-' * 10)


for epoch in range(epochs+1):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    #--- train ---
    train_loss_sum = 0
    train_total = 0
    train_correct = 0
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label_one_hot = torch.zeros(label.size(0),2).scatter_(1,label.unsqueeze(-1),1).to(device)
        label = label.to(device)

        out = resnet(data)
        out = classifier(out)

        train_loss = loss_fn(out, label_one_hot)
        train_loss_sum += train_loss

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        _, predicted = torch.max(out, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum()

    train_accuracy = 100 * train_correct / train_total
    print(f"train_loss : {train_loss_sum}, train_acc: {train_accuracy}")

    #--- validation ---
    val_loss_sum = 0
    val_total = 0
    val_correct = 0
    for data, label in tqdm(val_loader):

        with torch.no_grad():
            data = data.to(device)
            label_one_hot = torch.zeros(label.size(0),2).scatter_(1,label.unsqueeze(-1),1).to(device)
            label = label.to(device)


            out = resnet(data)
            out = classifier(out)

            val_loss = loss_fn(out, label_one_hot)
            val_loss_sum += val_loss

        _, predicted = torch.max(out, 1)
        val_total += label.size(0)
        val_correct += (predicted == label).sum()

    val_accuracy = 100 * val_correct / val_total
    print(f"val_loss : {val_loss_sum}, val_accuracy: {val_accuracy}")

    if epoch % 1 == 0:
        torch.save(classifier.state_dict(), f"./checkpoints/classifier_{epoch}.pth")

        

