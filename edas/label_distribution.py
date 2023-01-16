import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    annotation_file = "../data/task1/train_data/annotations.csv"
    annotations = pd.read_csv(annotation_file)

    # Plot label distribution
    labels = annotations["label"].values
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()