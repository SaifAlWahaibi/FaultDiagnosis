
# Python libraries: -

import torch

import Data_Preprocessing_v_2

import torch.nn as nn

from torchsummary import summary

import torch.optim as optim

from sklearn.metrics import confusion_matrix

import pandas as pd

import numpy as np

import copy

import random

# Seeding to ensure reproducibility: -

random.seed(0)

torch.manual_seed(0)

# Assigning device to run code on: -

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading training and testing data and their labels: -

training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels = \
    Data_Preprocessing_v_2.dataprepro()


# Data class for dataloader: -

class Dataset:

    def __init__(self, data, labels):

        self.input = torch.from_numpy(data)

        self.labels = torch.from_numpy(labels)

        self.size = data.shape[0]

    def __getitem__(self, index):

        img = self.input[index]

        img = img.view(1, 52, 52)

        return img.float(), self.labels[index]

    def __len__(self):

        return self.size


# Creating a data classes for the datasets: -

Training = Dataset(training_data, training_labels)

Validation = Dataset(validation_data, validation_labels)

Testing = Dataset(testing_data, testing_labels)


# Local and Global layer: -

class Local(nn.Module):

    def __init__(self, channel_in, channel_out, kernel):

        super().__init__()

        self.channel_in, self.channel_out, self.kernel = channel_in, channel_out, kernel

        self.Conv1 = nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_out,
                               kernel_size=(self.kernel, self.kernel), stride=(1, 1), padding=1)

        self.BatchNorm = nn.BatchNorm2d(channel_out)

        self.Act = nn.ReLU()

    def forward(self, x):

        out = self.Act(self.BatchNorm(self.Conv1(x)))

        return out


class Global(nn.Module):

    def __init__(self, channel_in, channel_out, height, width):

        super().__init__()

        self.channel_in, self.channel_out, self.height, self.width = channel_in, channel_out, height, width

        self.Conv1 = nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_out, kernel_size=(height, 1),
                               stride=(1, 1), padding=0)

        self.Conv2 = nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_out, kernel_size=(1, width),
                               stride=(1, 1), padding=0)

        self.BatchNorm = nn.BatchNorm2d(channel_out)

        self.Act = nn.ReLU()

    def forward(self, x):

        out = self.Act(self.BatchNorm(torch.matmul(self.Conv2(x), self.Conv1(x))))

        return out


# Model: -

writer = pd.ExcelWriter('LG-CNN_version_7_Test.xlsx')

k = 0

structure = [16]

for i in structure:

    class CNN(nn.Module):

        def __init__(self):

            super(CNN, self).__init__()

            self.Local = Local(1, i, 3)

            self.Global = Global(1, i, 52, 52)

            self.Pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

            self.Conv1 = nn.Conv2d(in_channels=2 * i, out_channels=4 * i, kernel_size=(3, 3), stride=(1, 1), padding=1)

            self.BatchNorm1 = nn.BatchNorm2d(4 * i)

            self.Act = nn.ReLU()

            self.Conv2 = nn.Conv2d(in_channels=4 * i, out_channels=8 * i, kernel_size=(3, 3), stride=(1, 1), padding=1)

            self.BatchNorm2 = nn.BatchNorm2d(8 * i)

            self.Conv3 = nn.Conv2d(in_channels=8 * i, out_channels=16 * i, kernel_size=(3, 3), stride=(1, 1), padding=1)

            self.BatchNorm3 = nn.BatchNorm2d(16 * i)

            self.Pool2 = nn.AvgPool2d(kernel_size=(26, 26), stride=(1, 1))

            self.Dense = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(16 * i, 20)
            )

        def forward(self, x):

            outA = self.Local(x)

            outB = self.Global(x)

            out = torch.cat((outA, outB), dim=1)

            out = self.Pool1(out)

            out = self.Act(self.BatchNorm1(self.Conv1(out)))

            out = self.Act(self.BatchNorm2(self.Conv2(out)))

            out = self.Act(self.BatchNorm3(self.Conv3(out)))

            out = self.Pool2(out)

            out = self.Dense(out)

            return out

    # Saving the model: -

    model = CNN()

    model = model.to(device)

    # Setting the model to training mood: -

    model.train()

    # Model architecture summary: -

    print(summary(model, (1, 52, 52), verbose=0))

    # Learning rate: -

    learning_rate = 1E-3

    # Optimizer: -

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function: -

    loss_fn = nn.CrossEntropyLoss()

    # Training dataloader: -

    TrBat = torch.utils.data.DataLoader(Training, batch_size=128, shuffle=True)

    # Validation dataloader: -

    ValBat = torch.utils.data.DataLoader(Validation, batch_size=128, shuffle=True)

    # Testing dataloader: -

    TeBat = torch.utils.data.DataLoader(Testing, batch_size=128, shuffle=True)

    # Training accuracy: -

    training_accuracy = []

    # Validation accuracy: -

    validation_accuracy = []

    # Cost: -

    cost = []

    # Run number: -

    k = k + 1

    # Epoch counter: -

    epoch = 0

    # An indicator for when the model is has stopped: -

    done = False

    # Learning loop: -

    while epoch < 1000 and not done:

        epoch += 1

        # Learning loop for each batch: -

        for imgs, labels in TrBat:

            # Saving images and labels in device: -

            imgs = imgs.to(device)

            labels = labels.to(device)

            # Running the model: -

            outputs = model(imgs)

            # Calculating the loss function: -

            loss = loss_fn(outputs, labels)

            # Backpropagation: -

            optimizer.zero_grad()

            loss.backward()

            # Updating parameters: -

            optimizer.step()

        cost.append(float(loss))

        print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

        # Calculating the training accuracy: -

        # Correctly predicted samples: -

        correct = 0

        # Total number of samples: -

        total = 0

        # Evaluating the model: -

        with torch.no_grad():

            for imgs, labels in TrBat:

                # Saving images and labels in device: -

                imgs = imgs.to(device)

                labels = labels.to(device)

                # Running the model: -

                outputs = model(imgs)

                outputs = torch.nn.functional.softmax(outputs, dim=1)

                # the model predictions: -

                _, predicted = torch.max(outputs, dim=1)

                # Calculating accuracy: -

                total += labels.shape[0]

                correct += int((predicted == labels).sum())

        training_accuracy.append(correct / total)

        # Calculating the validation accuracy: -

        correct = 0

        total = 0

        # Saving the prediction and the labels to construct a confusion matrix: -

        predictions = []

        truth = []

        # Evaluating the model on the validation set: -

        model.eval()

        with torch.no_grad():

            for imgs, labels in ValBat:

                imgs = imgs.to(device)

                labels = labels.to(device)

                outputs = model(imgs)

                outputs = torch.nn.functional.softmax(outputs, dim=1)

                _, predicted = torch.max(outputs, dim=1)

                total += labels.shape[0]

                correct += int((predicted == labels).sum())

                # Saving information for the confusion matrix: -

                predictions.extend(predicted.cpu().numpy())

                truth.extend(labels.cpu().numpy())

        validation_accuracy.append(correct / total)

        model.train()

        acc = correct / total

        # Early stopping: -

        # First epoch: -

        if epoch == 1:

            # Initializing counter: -

            counter = 0

            # First copies: -

            best_acc = acc

            best_model = copy.deepcopy(model)

            best_predictions = copy.deepcopy(predictions)

            best_truth = copy.deepcopy(truth)

        elif best_acc - acc < 0:

            # Re-initializing counter: -

            counter = 0

            # Saving new better performance model and the prediction & labels for the confusion matrix: -

            best_acc = acc

            best_model.load_state_dict(model.state_dict())

            best_predictions = copy.deepcopy(predictions)

            best_truth = copy.deepcopy(truth)

        elif best_acc - acc > 0.01:

            # Updating counter when performance deteriorates: -

            counter += 1

            # Existing epoch loop when patience condition is met: -

            if counter >= 10:

                model.load_state_dict(best_model.state_dict())

                torch.save(model.state_dict(), 'LG-CNN_version_7_Test_{}.pth'.format(k))

                done = True

    # Evaluating the model on the test set: -

    model.eval()

    correct = 0

    total = 0

    predictions = []

    truth = []

    with torch.no_grad():

        for imgs, labels in TeBat:

            imgs = imgs.to(device)

            labels = labels.to(device)

            outputs = model(imgs)

            outputs = torch.nn.functional.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, dim=1)

            total += labels.shape[0]

            correct += int((predicted == labels).sum())

            # Saving information for the confusion matrix: -

            predictions.extend(predicted.cpu().numpy())

            truth.extend(labels.cpu().numpy())

    model.train()

    # Calculating confusion matrix: -

    confusion = confusion_matrix(truth, predictions)

    # Printing results: -

    print("Confusion Matrix:")

    print(confusion)

    Confusion = pd.DataFrame(confusion)

    faults = 20

    fault_diagnosis_rate = []

    precision = []

    Fscore = []

    # Calculating FDR, precision and F1 score: -

    for j in range(0, faults):

        FDR = confusion[j, j] / np.sum(confusion[j, :])

        fault_diagnosis_rate.append(FDR)

        P = confusion[j, j] / np.sum(confusion[:, j])

        precision.append(P)

        F = (2 * P * FDR) / (P + FDR)

        Fscore.append(F)

    fault_diagnosis_rate = np.array(fault_diagnosis_rate)

    Fault_Diagnosis_Rate = pd.DataFrame(fault_diagnosis_rate)

    precision = np.array(precision)

    Precision = pd.DataFrame(precision)

    Fscore = np.array(Fscore)

    FScore = pd.DataFrame(Fscore)

    print("Fault Diagnosis Rate:")

    print(fault_diagnosis_rate.tolist())

    print("Average Fault Diagnosis Rate:")

    print(np.mean(fault_diagnosis_rate))

    print("Precision:")

    print(precision.tolist())

    print("Average Precision:")

    print(np.mean(precision))

    print("F1 Score:")

    print(Fscore.tolist())

    print("Average F1 Score:")

    print(np.mean(Fscore))

    print("Training Size:")

    print(training_labels.shape[0])

    print("Validation Size:")

    print(validation_labels.shape[0])

    print("Testing Size:")

    print(testing_labels.shape[0])

    print("-----------------------------------------------------------------------------------------------------------")

    Training_Accuracy = pd.DataFrame(training_accuracy)

    Training_Accuracy.to_excel(writer, sheet_name='Training Accuracy {}'.format(k))

    Validation_Accuracy = pd.DataFrame(validation_accuracy)

    Validation_Accuracy.to_excel(writer, sheet_name='Validation Accuracy {}'.format(k))

    Cost = pd.DataFrame(cost)

    Cost.to_excel(writer, sheet_name='Cost {}'.format(k))

    confusion = confusion_matrix(best_truth, best_predictions)

    Confusion = pd.DataFrame(confusion)

    Confusion.to_excel(writer, sheet_name='Confusion Matrix {}'.format(k))

    import numpy as np

    faults = 20

    fault_diagnosis_rate = []

    precision = []

    Fscore = []

    for j in range(0, faults):

        FDR = confusion[j, j] / np.sum(confusion[j, :])

        fault_diagnosis_rate.append(FDR)

        P = confusion[j, j] / np.sum(confusion[:, j])

        precision.append(P)

        F = (2 * P * FDR) / (P + FDR)

        Fscore.append(F)

    fault_diagnosis_rate = np.array(fault_diagnosis_rate)

    Fault_Diagnosis_Rate = pd.DataFrame(fault_diagnosis_rate)

    Fault_Diagnosis_Rate.to_excel(writer, sheet_name='FDR {}'.format(k))

    precision = np.array(precision)

    Precision = pd.DataFrame(precision)

    Precision.to_excel(writer, sheet_name='Precision {}'.format(k))

    Fscore = np.array(Fscore)

    FScore = pd.DataFrame(Fscore)

    FScore.to_excel(writer, sheet_name='F1 Score {}'.format(k))

writer.save()
