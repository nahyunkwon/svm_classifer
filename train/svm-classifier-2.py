from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize

from joblib import dump, load


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

    #flat_data = np.array(flat_data)
    #target = np.array(target)

    np.savetxt("./flat_data.csv", flat_data, delimiter=",")
    np.savetxt("./target.csv", target, delimiter=",")

    #images-stringing-support = np.array(images-stringing-support)
    #print(target)
    return Bunch(data=flat_data,
                 target=target,
                 #images-stringing-support=images-stringing-support,
                 target_names=categories,
                 DESCR=descr)


if __name__ == "__main__":
    #image_dataset = load_image_files("test-images-stringing-support/")



    X_train, X_test, y_train, y_test = \
        train_test_split(image_dataset.data, image_dataset.target, test_size=0.3, random_state=1)

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))

    dump(clf, './model/stringing_support_linear_1.joblib')

    '''

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    svc = svm.SVC()
    clf = GridSearchCV(svc, param_grid)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))
    '''



