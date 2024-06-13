import torch
import torchvision
import torchvision.transforms as transforms
import pickle

import matplotlib.pyplot as plt
import numpy as np

import claffierNN

import torch.optim as optim

import os

from PIL import Image

transform = transforms.Compose( [
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes= ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def imshow(images, labels, predicted):
    images = images / 2 + 0.5  # unnormalize
    np_images = images.numpy()

    fig, axes = plt.subplots(4, 8, figsize=(15, 8))  # Create a 4x8 grid of subplots
    axes = axes.flatten()
    for img, ax, label in zip(np_images, axes, labels):
        ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.set_title(label)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def displayer():
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(images, [classes[labels[j]] for j in range(batch_size)])
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


net = claffierNN.Net()

PATH = './cifar_net.pth'


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path= self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

def trainingSaving(device):
    net.TrainingLoss(trainloader, optim.SGD(net.parameters(), lr=0.001, momentum=0.9), device)
    torch.save(net.state_dict(), PATH)

# trainingSaving(device)
net.load_state_dict(torch.load(PATH))

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

labelsPredicted = []
whatAreTheImages = []

with torch.no_grad():

    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
       
        for label, prediction in zip(labels, predicted):
            labelsPredicted.append(classes[prediction])
            whatAreTheImages.append(classes[label])
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
for classname, correct_count in correct_pred.items():
    accuracy = 100 * correct_count / total_pred[classname]
    print(f'Accuracy for {classname} : {accuracy}%')


dataiter = iter(testloader)
images, labels = next(dataiter)
predicted_labels = [labelsPredicted[j] for j in range(batch_size)]

a = 0
for i in predicted_labels:
    if a%8 == 0:
        print()
    print(i, end=' ')
    a+= 1

imshow(images, [classes[labels[j]] for j in range(batch_size)], predicted_labels)
