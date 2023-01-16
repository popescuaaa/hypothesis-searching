from PIL import Image
import numpy as np
import os

if __name__ == '__main__':
    data_dir = "./data/task1/train_data/images/labeled/"
    # Count how many images we have
    images = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
    print("Number of images: ", images)