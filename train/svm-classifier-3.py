from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize

from joblib import dump, load

import pandas as pd


def load_image_files(container_path, dimension=(64, 64)):
    """
    Load image files with categories as subfolder names
    which performs like scikit-learn sample dataset

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to

    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []

    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            try:
                if img.shape[2] == 3:
                    print(file)
                    img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
                    print(img_resized.flatten().shape)
                    flat_data.append(img_resized.flatten())
                    images.append(img_resized)
                    target.append(i)
            except IndexError:
                pass

    # flat_data = np.array(flat_data)
    # target = np.array(target)

    np.savetxt("./data/stringing-cropped/flat_data.txt", flat_data)
    np.savetxt("./data/stringing-cropped/target.txt", target)

    # images-stringing-support = np.array(images-stringing-support)
    # print(target)


def train():
    flat_data = np.loadtxt("./data/stringing-cropped/flat_data.txt")
    target = np.loadtxt("./data/stringing-cropped/target.txt")
    # print(flat_data)

    X_train, X_test, y_train, y_test = \
        train_test_split(flat_data, target, test_size=0.3, random_state=0)

    # Create a svm Classifier
    clf = svm.SVC(kernel='rbf', gamma='scale')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print(len(X_test))

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))

    dump(clf, './model/stringing-cropped/stringing_cropped.joblib')


if __name__ == "__main__":
    load_image_files("./stringing_cropped")

    train()









