import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import torch.nn as nn
import torch
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

if __name__ == '__main__':
    labeled_images = "../data/task1/train_data/images/labeled/"
    unlabeled_images = "../data/task1/train_data/images/unlabeled/"
    annotation_file = "../data/task1/train_data/annotations.csv"

    annotations = pd.read_csv(annotation_file)
    # Train a classifier on the labeled images

    transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
    ])

    all_images = []
    all_labels = []
    for index, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Loading labeled images"):
        image = Image.open("../data/" + row["sample"])
        image = transforms(image)
        image = np.array(image)
        image = image.flatten()
        all_images.append(image)
        all_labels.append(row["label"])

    print("Number of labeled images: ", len(all_images))
    print(all_images[0].shape)

    # Train a classifier on the labeled images
    # The best knn metric for images is cosine
    knn = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(all_images))), metric="cosine")
    knn.fit(all_images, all_labels)

    # Predict the labels of the unlabeled images
    unlabeled_images = os.listdir(unlabeled_images)
    unlabeled_images = [transforms(Image.open("../data/task1/train_data/images/unlabeled/" + image)) for image in tqdm(unlabeled_images, total=len(unlabeled_images), desc="Loading unlabeled images")]
    unlabeled_images = [np.array(image).flatten() for image in tqdm(unlabeled_images, total=len(unlabeled_images), desc="Flattening unlabeled images")]
    print("Number of unlabeled images: ", len(unlabeled_images))
    print(unlabeled_images[0].shape)

    results = knn.predict(unlabeled_images)
    print(results)

    # Print the cosine distance one image with class 1 from labeled and one image with class 1 predicted
    target_class = 10
    l_image = None
    for i in range(len(all_images)):
        if all_labels[i] == target_class:
            print("Labeled image with class 1: ", all_images[i])
            l_image = all_images[i]
            break

    u_image = None
    for i in range(len(unlabeled_images)):
        if results[i] == target_class:
            print("Unlabeled image with class 1: ", unlabeled_images[i])
            u_image = unlabeled_images[i]
            break

    # show images
    plt.imshow(l_image.reshape(64, 64), cmap='YlGnBu')
    plt.show()
    plt.imshow(u_image.reshape(64, 64), cmap='YlGnBu')
    plt.show()

    # Create a new annotation file with the predicted labels + the labeled images
    new_annotations = pd.DataFrame(columns=["sample", "label"])
    unlabeled_images = os.listdir("../data/task1/train_data/images/unlabeled/")
    for i in range(len(unlabeled_images)):
        new_annotations = new_annotations.append({"sample": "task1/train_data/images/unlabeled/" + unlabeled_images[i], "label": results[i]}, ignore_index=True)

    new_annotations = new_annotations.append(annotations, ignore_index=True)
    new_annotations.to_csv("../data/task1/train_data/full_annotations.csv", index=False)
