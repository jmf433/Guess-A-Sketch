import sys

import torch
import math
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import matplotlib.pyplot as plt
import cv2


class DrawingsDataset(Dataset):
    """A pyTorch dataset containing the vector versions of the drawings."""

    def __init__(self, location='TrainingData', n=80, transform=None):
        """
        Inputs:
            location (string): Parent directory with all the images (either 'OriginalData' or 'TrainingData').
                location must subdirectories for each class. Images in this location must be numbered alphabetically
                by class (within class order doesn't matter as long as there are no gaps in numbering).
            n (integer): the number of images per class. In the default dataset, n=80

        Properties:
            names (array): list of all unique class names
        """
        location = os.path.join(os.getcwd(), location)
        self.names = sorted(os.listdir(location))
        self.n = n
        self.location = location

    def __len__(self):
        """
        Returns:
            total number of images
        """
        return len(self.names)*self.n

    def __getitem__(self, idx):
        """
        Inputs:
            idx (integer): the index of the image to retrieve
        Returns:
            a tuple containing (the label, a numpy array of the loaded image)
        """
        idx = idx+1
        label = self.names[math.floor((idx-1)/self.n)]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # retrieves path to the images indicated by idx
        img_name = os.path.join(self.location,
                                label,
                                str(idx) + '.png')
        # reads the images
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        return (label, image)


def extractData(batch_size, type, location='256Train'):
    '''
    Returns batch_size images from a specific folder

    Inputs:
    batch_size (integer): the number of images to use in each training loop
    type (string): either 'training','validation', or 'testing'
    location (string): the location to retrieve images from

    Returns: a tuple (training, testing) containing a pyTorch Dataloaders
    '''
    shuffle = True

    # if we are using the 256Train/Test/Validation folders, split is 80/10/10
    n = 8
    if type == "training":
        n = 64

    # in the other folders, split is more toward testing
    if location == "NewTestingData":
        n = 5
    elif location == "NewTrainingData":
        n = 75

    training = DrawingsDataset(location, n, transform)
    trainingLoader = DataLoader(training, batch_size, shuffle)

    return (trainingLoader)
