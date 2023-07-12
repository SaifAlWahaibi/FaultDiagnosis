
# Python libraries: -

import pandas as pd

from sklearn import preprocessing

import numpy as np

# Function to pre-process data: -

def dataprepro():

    # Loading data: -

    training = pd.read_csv('training40t5v5t.csv')

    validation = pd.read_csv('validation40t5v5t.csv')

    testing = pd.read_csv('testing40t5v5t.csv')

    # Calculating the mean and standard deviation of the training dataset: -

    normalizing = preprocessing.StandardScaler().fit(training.iloc[:, 3:])

    # Normalizing the training dataset: -

    training.iloc[:, 3:] = normalizing.transform(training.iloc[:, 3:])

    # Normalizing the validation dataset (autoscaling): -

    validation.iloc[:, 3:] = normalizing.transform(validation.iloc[:, 3:])

    # Normalizing the testing dataset (autoscaling): -

    testing.iloc[:, 3:] = normalizing.transform(testing.iloc[:, 3:])

    # Number of samples in an image and step size: -

    r = 52

    s = 20

    # Collecting data and assigning labels: -

    training_labels = []

    training_data = []

    for i in range(0, training.shape[0], s):

        if training.shape[0] - i < r:

            continue

        data = training.iloc[i:(r + i), 3:].to_numpy()

        training_data.append(data)

        label = training[i:(r + i)].faultNumber.value_counts().idxmax()

        label = label - 1

        training_labels.append(label)

    training_data = np.asarray(training_data)

    training_labels = np.array(training_labels)

    validation_labels = []

    validation_data = []

    for i in range(0, validation.shape[0], s):

        if validation.shape[0] - i < r:

            continue

        data = validation.iloc[i:(r + i), 3:].to_numpy()

        validation_data.append(data)

        label = validation[i:(r + i)].faultNumber.value_counts().idxmax()

        label = label - 1

        validation_labels.append(label)

    validation_data = np.asarray(validation_data)

    validation_labels = np.array(validation_labels)

    testing_labels = []

    testing_data = []

    for i in range(0, testing.shape[0], s):

        if testing.shape[0] - i < r:

            continue

        data = testing.iloc[i:(r + i), 3:].to_numpy()

        testing_data.append(data)

        label = testing[i:(r + i)].faultNumber.value_counts().idxmax()

        label = label - 1

        testing_labels.append(label)

    testing_data = np.asarray(testing_data)

    testing_labels = np.array(testing_labels)

    return training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels

if __name__ == "__main__":

    training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels = dataprepro()
