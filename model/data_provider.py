from __future__ import print_function
import pickle
from os import path, listdir
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import scipy.misc
import numpy as np
import sys



class DataBatchProvider:

    batch_offset = 0
    file_offset = 0
    epochs_complete = 0
    images = {}
    labels = {}

    #TODO : make list
    #
    #   From image directory, obtain image file list and make them list
    #   and match the of the images and labels.
    #   Then, separate them into training set and test set.
    #
    def __init(self, mode, foo):
        print("Initializing DataBatchProvider..", end='')
        self.images = sorted(listdir(foo.img_batch_path))   # foo
        self.labels = sorted(listdir(foo.class_batch_path)) # foo


    #TODO : make mini-batch
    #
    #  From image directory, choose n images to argument randomly.
    #  Then aggregate the argumented images and make them batch.
    #  Note that all images is provided once in a epoch.
    #
    def load_batch(self):
        pass

    #TODO : argumentation
    #
    #  From a image with any size, argumenting n images.
    #  If size of the given image is less than or equal to the network required,
    #  Then do data argumentation instead of argumenting.
    #
    def argm(self):
        pass

    #TODO : pass next batch
    #
    #  Feed next batch made by load_barch function to network.
    #
    #
    def next_batch(self):
        pass

    def reset_batch_offset(self):
        self.batch_offset = 0
