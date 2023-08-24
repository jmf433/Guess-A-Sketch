import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import multiprocessing as mp
import pickle
from ProcessData import *

import matplotlib.pyplot as plt


class ConvNet(nn.Module):
    '''
    Initializes the neural network for image classification.
    Contains 5 convolutional layers and 1 fully connected layer, as well as
    2x2 max pooling layers after each convolutional layer. Saves the history
    of the loss over time.
    '''

    def __init__(self, output_dim):
        super(ConvNet, self).__init__()

        # initializing 5 layers of CNN
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(192, 384, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(384*16*16, output_dim)

    def forward(self, x):
        # Implement the forward pass, with ReLU non-linearities and max-pooling
        x = x.reshape(x.size(axis=0), 1, 256, 256)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = nn.functional.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


def train_nth_cnn(trainingLoader, validationLoader, word_length, nclasses,
                  label_lengths, train_labels, num_epochs, lr=1e-5):
    '''
    The code body for training the CNN model for a particular word length, called
    by train_nth_model

    Inputs:
        trainingLoader: data trainingLoader object
        validationLoader: data trainingLoader object for validation data
        word_length: the word length that this model is intended for
        nclasses: the number of unique classes of this length
        label_lengths: array of the lengths of labels in all_labels
        train_labels: tuple containing the unique labels of length word_length
        num_epochs: number of epochs
        lr: learning rate
    Outputs:
        conv_model: the trained model for this particular word length
        training_losses: a tensor of length num_epochs that stores the average 
            training loss per epoch
        validation_losses: a tensor of length num_epochs that stores the 
            validation loss per epoch
    '''
    # initialize CNN object
    conv_model = ConvNet(output_dim=nclasses)

    # fit a label encoder to all labels of this word length
    le = preprocessing.LabelEncoder()
    le.fit(train_labels)
    training_losses = torch.zeros(num_epochs)
    validation_losses = torch.zeros(num_epochs)

    # words with more labels (more training data) should have a larger learning
    # rate
    lr = lr/(1+1/len(train_labels))
    print("learning rate for length", word_length, "is", lr)

    optimizer = optim.SGD(conv_model.parameters(), lr=lr, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: 0.99)
    for epoch in range(num_epochs):
        epoch_training_loss = 0
        n = 0
        for batch, (temp_label, imgs) in enumerate(trainingLoader):

            train_labels = []
            label_lengths = torch.zeros(len(temp_label))

            # determine the lengths of all labels in this iteration of dataLoader
            for i in range(len(temp_label)):
                l = len(temp_label[i])
                label_lengths[i] = l
                # only keep labels of the appropriate length
                if l == word_length:
                    train_labels.append(temp_label[i])

            # transform the string labels to numeric using label encoder
            train_labels = torch.tensor(
                (le.transform(train_labels)), dtype=torch.long)

            # extract the images associated with labels of appropriate word length
            indicies = label_lengths == word_length
            # scale
            train_images = imgs[indicies]/255.0

            # skip iterations where no images had appropriate label length
            if torch.sum(indicies) == 0:
                continue

            # pass into CNN
            optimizer.zero_grad()
            preds = conv_model.forward(train_images)
            training_loss = nn.functional.cross_entropy(preds, train_labels)
            epoch_training_loss += training_loss.item()
            n += 1
            training_loss.backward()
            optimizer.step()
        scheduler.step()
        # store the average loss for this epoch
        if word_length == 3:
            print(train_labels)
            print(torch.argmax(preds, dim=1))

        ii = []
        valid_labels, valid_imgs = next(iter(validationLoader))
        subset_valid_labels = []
        for i in range(len(valid_labels)):
            if len(valid_labels[i]) == word_length:
                ii.append(i)
                subset_valid_labels.append(valid_labels[i])
        subset_valid_labels = torch.tensor(
            (le.transform(subset_valid_labels)), dtype=torch.long)
        subset_valid_imgs = valid_imgs[ii]/255.0

        preds = conv_model.forward(subset_valid_imgs)
        validation_loss = nn.functional.cross_entropy(
            preds, subset_valid_labels)
        validation_losses[epoch] = validation_loss.item()

        training_losses[epoch] = epoch_training_loss/n
        print("loss for Epoch", str(epoch),
              "and word length", word_length, "is", training_losses[epoch])

    return conv_model, training_losses, validation_losses


def train_nth_model(i, unique_lengths, counts, all_labels, trainingLoader,
                    validationLoader, label_lengths, num_epochs, lr):
    '''
    Trains the CNN model for a particular word length

    Inputs:
        i: The index of the word length in unique_lengths
        unique_lengths: the unique lengths of all labels
        counts: the counts of each length
        all_labels: tuple containing the set of all possible labels
        trainingLoader: data trainingLoader object
        trainingLoader: data trainingLoader object for validation data
        label_lengths: array of the lengths of labels in all_labels
        num_epochs: number of epochs
        lr: learning rate
    Outputs:
        word_length: the word length the model was trained for
        conv_model: the corresponding CNN model
        loss_history: the loss history of that model
    '''
    # extract the ith word length and its count
    word_length = int(unique_lengths[i].numpy())
    nclasses = int(counts[i].numpy())

    # find which unique labels are associated with this word length
    # note that we cannot vectorize as all_labels is still string labels and they
    # must be appended to a list
    train_labels = []
    for i in all_labels:
        if len(i) == word_length:
            train_labels.append(i)

    # train the model
    conv_model, train_loss, valid_loss = train_nth_cnn(trainingLoader, validationLoader, word_length, nclasses,
                                                       label_lengths, train_labels, num_epochs, lr)

    print("cnn for length " + str(word_length) + " trained")
    # add to the training loss
    with open("CNNTrainingLossHistory.obj", "rb") as f:
        train_loss_history = pickle.load(f)

    train_loss_history[word_length] = train_loss

    with open('CNNTrainingLossHistory.obj', 'wb') as f:
        pickle.dump(train_loss_history, f)

    # add to the validation loss

    with open("CNNValidationLossHistory.obj", "rb") as f:
        valid_loss_history = pickle.load(f)

    valid_loss_history[word_length] = valid_loss

    with open('CNNValidationLossHistory.obj', 'wb') as f:
        pickle.dump(valid_loss_history, f)

    return word_length, conv_model


def train(batch_size, num_epochs, lr=1e-5):
    '''
    Trains a CNN model using images in the "NewTrainingData" folder

    Inputs:
        batch_size: The number of images to train on in each epoch
        num_epochs: The number of epochs
        lr: The learning rate, default 1e-5
    Outputs:
        Saves the trained model to file 'CNNModel.obj'
        Saves the loss histories to file 'CNNTrainingLossHistory.obj' and 
            'CNNValidationLossHistory.obj'
    '''
    # init a file to log the loss
    with open("CNNTrainingLossHistory.obj", "wb") as f:
        pickle.dump({}, f)
    with open("CNNValidationLossHistory.obj", "wb") as f:
        pickle.dump({}, f)

    # use the dataLoader to extract training data
    trainingLoader = extractData(batch_size, "training", "256Train2")
    validationLoader = extractData(2000, "validation", "256Val2")

    # extract all unique labels by looking at the "NewTrainingData" directory
    all_labels = sorted(os.listdir("NewTrainingData"))

    # determine the lengths of each label
    label_lengths = torch.zeros(len(all_labels))
    for i in range(len(all_labels)):
        label_lengths[i] = len(all_labels[i])

    # determine the unique lengths, and the numbers of classes which have those lengths
    unique_lengths, counts = torch.unique(label_lengths, return_counts=True)
    res = {}

    # concurrently train models, one per label length
    pool = mp.Pool(6)
    results = pool.starmap(train_nth_model, [(
        i, unique_lengths, counts, all_labels, trainingLoader, validationLoader,
        label_lengths, num_epochs, lr)
        for i in range(len(unique_lengths))])

    # put the results in an array
    for length, model in results:
        res[float(length)] = model

    filehandler = open('CNNModel.obj', 'wb')
    pickle.dump(res, filehandler)
    return


def test(batch_size):
    '''
    Tests the model located in file 'CNNModel.obj' on testing images in the
    folder 'NewTestingData'

    Inputs:
        batch_size: The number of images to test on, maximum 1250
    Outputs:
        accuracy: The fractions of correctly classified images
    '''
    # extract a dataLoader for the testing data
    testingLoader = extractData(batch_size, "testing", "256Test2")
    test_labels, test_imgs = next(iter(testingLoader))

    # scale
    test_imgs = test_imgs/255.0

    # load the trained model
    filehandler = open('CNNModel.obj', 'rb')
    models = pickle.load(filehandler)

    # extract all possible labels
    all_labels = sorted(os.listdir("NewTrainingData"))
    all_label_lengths = torch.zeros(len(all_labels))
    for i in range(len(all_labels)):
        all_label_lengths[i] = len(all_labels[i])

    # determine the lengths of each label in testing data
    label_lengths = torch.zeros(len(test_labels))
    for i in range(len(test_labels)):
        label_lengths[i] = len(test_labels[i])

    # determine the unique lengths, and the numbers of classes which have those lengths
    unique_lengths, counts = torch.unique(label_lengths, return_counts=True)
    # evaluate each word on the model for their respective length
    labels = np.array(test_labels)
    # Compute the model accuracy
    ncorrect = 0
    ntotal = 0

    for i in range(len(unique_lengths)):
        word_length = int(unique_lengths[i].numpy())
        model = models[word_length]

        # generate a label encoder for these allowed labels
        selected_labels = []
        # need to use for loop because can't slice tuple of strings
        for l in all_labels:
            if len(l) == word_length:
                selected_labels.append(l)
        le = preprocessing.LabelEncoder()
        le.fit(selected_labels)

        # use this model on only the images with this label length
        indices = label_lengths == word_length
        imgs = test_imgs[indices]
        preds = model.forward(imgs)
        (pPred, preds) = torch.max(preds, 1)

        print("label vs prediction")
        print("word length is ", word_length)
        print(labels[indices])
        print(le.inverse_transform(preds))

        # transform the string testing labels to numeric
        new_labels = le.transform((labels[indices]))

        # increment the number of matching labels and total labels
        match = torch.sum(preds == torch.from_numpy(new_labels))
        match_count = (torch.sum(match)).item()
        ncorrect += match_count
        ntotal += len(new_labels)

    return ncorrect/ntotal


def plotLoss(word_length, type):
    '''
    Creates a plot of loss versus epoch for particular model.
    Inputs:
        word_length: the word length for which to plot the loss for
        type: either "training" or "validation"
    '''
    # load the trained model
    if type == "training":
        filehandler = open('CNNTrainingLossHistory.obj', 'rb')
    else:
        filehandler = open('CNNValidationLossHistory.obj', 'rb')
    losses = pickle.load(filehandler)
    loss_history = losses[word_length]

    plt.plot(range(len(loss_history)), loss_history)
    plt.title(type + " loss for word length " + str(word_length))
    plt.xlabel("Epoch")
    plt.ylabel("Average Cross Entropy Loss")
    plt.show()


if __name__ == '__main__':
    num_epochs = 50
    lr = 0.01

    # train(850, num_epochs, lr)
    accuracy = test(2000)
    print(accuracy)
    for i in range(3, 17):
        plotLoss(i, "training")
