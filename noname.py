import tensorflow as tf
import numpy as np
from model.model import *
from model.run import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 1e-6, "initial learning rate")
flags.DEFINE_boolean("debug", True, "activate verbose")
flags.DEFINE_string("mode", 'train', "train / test")
flags.DEFINE_int("crop", 10, "number of cropping")
flags.DEFINE_string("net", 'vgg16', "the architecture that will be used.")
flags.DEFINE_string("dataset", 'IMGNET2012', "dataset")

models = {
          'vgg16' : {'name' : 'vgg16','img_size' : [224,224,3], 'network' : VGG16net,
                     'weights_path' : None},
          'resnet' : {'name' : 'resnet', 'img_size' : []}
         }

function = {'train' : train, 'test' : test}

#TODO : set datasets
#
#   1. choose datasets
#   2. write tables for indices to labels name
#
dataset = {
          'IMAGENET2012' : {'name':'IMGNET2012', 'numofclass' : 1000, 'path': None,
                            'labels_table' : []},
          }

assert all(len(dataset[k]['labels_table'])>0 for k in dataset.keys()), "ERROR : There is no table for labels"
assert FLAGS.mode in ["train", "test"], "ERROR : Invaild mode."
assert FLAGS.net in models.keys(), "ERROR : Invalid network name."
assert FLAGS.dataset in dataset.keys(), "ERROR, Invalid dataset name."
assert FLAGS.learning_rate > 0, "ERROR : Invalid learning_rate."
assert FLAGS.crop > 0, "ERROR : INvalid cropping."

function[FLAGS.mode](models, FLAGS)
