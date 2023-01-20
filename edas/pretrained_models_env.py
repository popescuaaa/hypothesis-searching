from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import PIL
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import re

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, 0

def train_model(model, model_name, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, scheduler=None):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)




    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Save the model
    torch.save(model.state_dict(), "./noisy_models/{}_{}.pth".format(model_name, num_epochs))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet_features":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        # Without the last layer
        model_ft.fc = nn.Identity()
        input_size = 224

    elif model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained, progress=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


if __name__ == '__main__':
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    print("Task 1: Missing labels")

    # Fix seed
    torch.manual_seed(0)
    np.random.seed(0)

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    training_data_dir = "../data/task2/classes_train"
    validation_data_dir = "../data/task2/classes_val"

    val_data_dir = "../data/task2/val_data"
    annotation_file = "../data/task2/train_data/annotations.csv"

    # Models to choose from [resnet50, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet50"

    # Number of classes in the dataset
    num_classes = 100

    # Batch size for training (change depending on how much memory you have)
    batch_size = 8

    # Number of epochs to train for
    num_epochs = 10

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Learning rate for training
    learning_rate = 1e-3

    # Use scheduler
    use_scheduler = True

    # Use probabilistic sampling
    use_probabilistic_sampling = False

    for model_name in ["resnet"]:
        print("Model: ", model_name)

        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

        # Print the model we just instantiated
        print(model_ft)

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        image_datasets = {
                "train": datasets.ImageFolder(training_data_dir, data_transforms["train"]),
                "val": datasets.ImageFolder(validation_data_dir, data_transforms["val"])
        }

        # Create training and validation dataloaders
        dataloaders_dict = {
                "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=4),
                "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=4)
        }

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)

        # Scheduler
        if use_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        else:
            scheduler = None

        # Set up the loss fxn
        if use_probabilistic_sampling:
            df = pd.read_csv(annotation_file)
            class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
            class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        else:
            criterion = nn.CrossEntropyLoss(reduction='mean')

        # Train and evaluate
        model_ft, hist = train_model(model_ft, model_name, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                     is_inception=(model_name == "inception"), scheduler=scheduler)


        # Load validation images and apply the same transformations as the training images
        validation_images = os.listdir(val_data_dir)
        image_names = copy.deepcopy(validation_images)

        # Sort the images by the numbers in their names
        validation_images.sort(key=lambda f: int(re.sub('\D', '', f)))

        validation_images = [os.path.join(val_data_dir, image) for image in validation_images]
        validation_images = [Image.open(image) for image in validation_images]
        validation_images = [img.convert('RGB') for img in validation_images]
        validation_images = [data_transforms["val"](image) for image in validation_images]

        model_ft.eval()
        submission_file_name = './noisy_submissions/{}_{}_{}_{}_{}.csv'.format(model_name,
                                                  num_epochs,
                                                  learning_rate,
                                                  "use_scheduler_{}".format(use_scheduler),
                                                  "prob_samp_{}".format(use_probabilistic_sampling))
        with torch.no_grad():
            with open(submission_file_name, 'w') as f:
                f.write("sample,label\n")
                for i, image in tqdm(enumerate(validation_images), total=len(validation_images), desc="Predicting"):
                    image = image.to(device)
                    image = image.unsqueeze(0)
                    outputs = model_ft(image)
                    _, preds = torch.max(outputs, 1)
                    f.write("{},{}\n".format(image_names[i], preds.item()))

        submission = pd.read_csv(submission_file_name)
        # Order the submission by the numbers in the sample name
        submission["sample"] = submission["sample"].apply(lambda x: int(re.findall(r'\d+', x)[0]))
        submission = submission.sort_values(by="sample")
        submission["sample"] = submission["sample"].apply(lambda x: str(x) + ".jpeg")
        submission.to_csv(submission_file_name, index=False)




