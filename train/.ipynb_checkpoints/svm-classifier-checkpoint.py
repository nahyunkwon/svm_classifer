import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
#%matplotlib inline

import pandas as pd
import numpy as np

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc


def get_image(row_id, root="./images-stringing-support/"):
    """
    Converts an image number into the file path where the image is located,
    opens the image, and returns the image as a numpy array.
    """
    filename = "{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)


def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2gray(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    return flat_features


def create_feature_matrix(label_dataframe):
    features_list = []

    for img_id in label_dataframe.index:
        # load image
        img = get_image(img_id)
        # get features for image
        image_features = create_features(img)
        features_list.append(image_features)

    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix


if __name__ == "__main__":
    labels = pd.read_csv("./stringing-1.csv", index_col=0)
    print(labels)

    # run create_feature_matrix on our dataframe of images-stringing-support
    feature_matrix = create_feature_matrix(labels)

    # get shape of feature matrix
    print('Feature matrix shape is: ', feature_matrix.shape)

    # define standard scaler
    ss = StandardScaler()
    # run this on our feature matrix
    bees_stand = ss.fit_transform(feature_matrix)

    pca = PCA(n_components=500)
    # use fit_transform to run PCA on our standardized matrix
    bees_pca = ss.fit_transform(bees_stand)
    # look at new shape
    print('PCA matrix shape is: ', bees_pca.shape)

    X = pd.DataFrame(bees_pca)
    y = pd.Series(labels.genus.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234123)

    # look at the distrubution of labels in the train set
    pd.Series(y_train).value_counts()

    # define support vector classifier
    svm = SVC(kernel='linear', probability=True, random_state=42)

    # fit model
    svm.fit(X_train, y_train)

    # generate predictions
    y_pred = svm.predict(X_test)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model accuracy is: ', accuracy)

    # generate predictions
    y_pred = svm.predict(X_test)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model accuracy is: ', accuracy)