import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
    annotation_file = "../data/task2/train_data/annotations.csv"
    annotations = pd.read_csv(annotation_file)

    # Create a folder with class folders based on the annotations

    # Solit images based on their classes to create a balanced dataset
    frequencies = [0] * 100
    for index, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Counting frequencies"):
        frequencies[row["label_idx"]] += 1

    # Create a folder for each class
    for i in range(len(frequencies)):
        # Train
        if not os.path.exists("../data/task2/classes_train/" + str(i)):
            os.makedirs("../data/task2/classes_train/" + str(i))

        # Val
        if not os.path.exists("../data/task2/classes_val/" + str(i)):
            os.makedirs("../data/task2/classes_val/" + str(i))

    # Split the images into train and val
    for index, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Splitting images"):
        if np.random.rand() < 0.8:
            os.system("cp ../data/" + row["renamed_path"] + " ../data/task2/classes_train/" + str(row["label_idx"]) + "/")
        else:
            os.system("cp ../data/" + row["renamed_path"] + " ../data/task2/classes_val/" + str(row["label_idx"]) + "/")

    # Count the number of images per class
    class_counts = {}
    for index, row in annotations.iterrows():
        if row["label_idx"] in class_counts:
            class_counts[row["label_idx"]] += 1
        else:
            class_counts[row["label_idx"]] = 1

    frequencies = [class_counts[i] for i in class_counts]
    print(frequencies)
