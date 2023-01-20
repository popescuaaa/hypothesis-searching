# Contrastive loss training for Resent50 pretrained on ImageNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import torchvision
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import PIL

# Use resnet50 pretrained in a contrastive learning classification task with encoder and custom loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 100)

    def forward(self, x):
        x = self.resnet(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.encoder = Encoder()

    def forward(self, x1, x2):
        output1 = self.encoder(x1)
        output2 = self.encoder(x2)
        return output1, output2

class SiameseNetworkDataset(torch.utils.data.Dataset):
    def __init__(self, imageFolderDataset: torch.utils.data.Dataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset)


if __name__ == '__main__':
    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root="../data/task1/classes_train", transform=transform)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=dataset, transform=transforms, should_invert=False)
    dataset_loader = torch.utils.data.DataLoader(siamese_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Create the model
    model = SiameseNetwork()
    model = model.cuda()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Train the model with positive and negative pairs
    for epoch in range(10):
        for i, data in tqdm(enumerate(dataset_loader, 0), total=len(dataset_loader)):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))

    # Save the model
    torch.save(model.state_dict(), "./models/siamese.pth")

    # Test the model on an image to predict the class
    model = SiameseNetwork()
    model.load_state_dict(torch.load("./models/siamese.pth"))
    model = model.cuda()

    # Load the image
    test_image = dataset_loader.dataset[0][0].unsqueeze(0).cuda()
    test_image = test_image.cuda()

    # Predict the class
    model.eval()
    with torch.no_grad():
        output = model.encoder(test_image)
        _, preds = torch.max(output, 1)
        print(preds.item())


