from __future__ import print_function
import tensorflow as tf
import time
import numpy as np
import math
import libmr
import urllib.request
import os
from abc import *
import sys
mean = {}

def printProgress(current, total, prefix='', suffix='', barLength=100):
    percent = current*100 / float(total)
    assert percent <= 100 and percent >= 0, "Invaild total and current"
    progressed = int(percent * barLength / 100)
    bar = '#' * progressed + '-' * (barLength-progressed)
    sys.stdout.write('\r%s |%s| %s%s %s'%(prefix, bar, percent, '%', suffix))
    if current == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def time_cal(time):
    rt = ''
    T = {'h':0, 'm':0, 's':0}
    sl = 3600
    for elem in T:
        T[elem] = int(time/sl)
        time -= T[elem]*sl
        sl /= 60
        if T[elem]:
            rt += ("%d %s " %(T[elem], elem))
    return rt

class network(metaclass = ABCMeta):

    def __init__(self, x,y, numofclass,sess, weights_path,  log_dir, verbose):
        self.X = x # input
        self.Y = y # label
        self.numofclass = numofclass
        self.verbose = verbose
        self.weights_path = weights_path
        self.log_dir = log_dir
        self.checkpoint(sess)
        self.param = []
    def checkpoint(self, sess): # To determine loading weights or checkpoint
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if self.verbose : print("Restoring checkpoint..", end='')
            sess.restore(sess, "")
            if self.verbose : print("Done")
        else:
            self.load_weights(sess)

    #TODO : openMax
    #
    #   ALGORITHM #2
    #
    def openMax(self, sess):
        pass

    #TODO : Meta-Recognition
    #
    # get Meta-Recognition score.
    #
    @abstractmethod
    def get_meta_logits(self):
        pass

    def load_weights(self, sess): ## load pre-trained network weights
        if self.verbose : print("Loading pretrained model..", end='')
        weights = np.load(weight_path)
        weights_key = sorted(weights_key())
        for i, k in enumerate(weights_key):
            if i == len(self.param):
                break
            if self.verbose:
                print("The shape of %s of the pretrained model is %s."
                      " And the shape of %s of the model is %s"
                      %(k, weights[k].shape, k, self.param[i].shape))
            sess.run(self.param[i].assign(weights[k].reshape(self.param[i].shape)))
        if self.verbose: print("Done.")

    @abstractmethod
    def get_logits(self): #return logits
        pass

    @abstractmethod #from pre-trained layers, obtain feature.
    def get_feature(self):
        pass

    #TODO : get weight file
    #
    #   Find weights file from the certain directory.
    #   IF not being able to find it and URL for weight file is given,
    #   download the file in the certain folder.
    #
    def get_model(self,url):

        model_path = os.getcwd()
        weight_path = './weights'
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)

        os.chdir(weight_path)

        file_name = url.split('/')[-1]

        u = urllib.request.urlopen(url)
        file_meta = u.info()
        file_size = int(file_meta["Content-Length"])

        if not os.path.exists(file_name) or os.stat(file_name).st_size != file_size:
            if self.verbose : print("There is no pretrained model.")
            start_time = time.time()
            with open(file_name, 'wb') as f:
                print("Download: %s Bytes: %s" % (file_name, file_size))
                file_size_dl = 0 # downloaded
                file_block_size = 1024 * 64 # 64kb
                while True:
                    buffer = u.read(file_block_size)
                    if not buffer:
                        break
                    file_size_dl += len(buffer)
                    f.write(buffer)
                    printProgress(file_size_dl, file_size, 'Progress', 'Complete', 20)
            print("Download complete. it takes %s " %(time_cal(int(time.time()-start_time))))
        else:
            print("There is pretrained model already.")

        return os.path.join(os.getcwd(), file_name)


    """

    for debugging (with tensorboard)

    """

    def add_activation_summary(var):
        if var is not None:
            tf.summary.histogram(var.op.name + "/activation", var)
            tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


    """

    define layers for simplicity

    """
    def _max_pool2_(self, x, name):
        """
            input
            - x : feature

            Output
            - feature max_pooled with strides 2 and "SAME" zero padding
        """
        return tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'SAME', name=name)

    def init_variable(self,name, weights_shape, biases_shape, params, pretrained, trainable):
        with tf.variable_scope(name) as scope:
            w_init = tf.truncated_normal(shape = weights_shape, stddev=math.sqrt(2.0)/(params))
            b_init = tf.constant(0.0, shape=biases_shape)
            weights = tf.get_variable('weights', initializer = w_init, trainable = trainable)
            biases = tf.get_variable('biases', initializer = b_init, trainable = trainable)
            if pretrained:
                self.param.append(weights)
                self.param.append(biases)
            return weights, biases

    def fc(self, x, output_dim, name, relu = True, dropout=True, drop_prob=0.5, pretrained =True, trainable=True):
        """
            Inputs
            - x : input feature with [N, H, W, C]
            - output_dim : output dimension
            - name : name for fully connected layer
            Outputs
            - a matrix with [N, output_dim]
        """
        shape = x.get_shape().as_list()
        x_reshaped = tf.reshape(x, [-1, (np.prod(np.array(shape[1:])))])
        reshape_shape = x_reshaped.get_shape().as_list()
        weights_shape = [reshape_shape[-1], output_dim]
        biases_shape = [output_dim]
        weights, biases = init_variable(name, weights_shape, biases_shape, tf.size(weights_shape), pretrained, trainable)
        with tf.variable_scope(name) as scope:
            out = tf.matmul(x_reshaped, weights)
            out = tf.nn.bias_add(out, biases)
            if relu:
                out = tf.nn.relu(out, name = name+'_relu')
            if dropout:
                out = tf.nn.dropout(out, drop_prob)
        return out

        out = tf.matmul()
    def conv(self, x, num_filters, name, drop_prob=0.5, strides=1, padding='SAME', repeat=1, pretrained = True, ks=3, relu=True, dropout=False, trainable=False):
        """
            Inputs
            - x : input feature
            - num_filters : # of filters
            - ks : kernel size
            - repeat : # of conv layers which repeat under same parameter(e.g. strides, num_filters, kernel size, and so on)
            - name : name for layer(s)
            - pretrained : Is there pre-trained weights for this layer(s).
            - trainable : Is this layer(s) trainable.

            Outputs
            - activation
        """

        out = x
        param = [] # for load_weights function, it includes parameters to load weights
        scopes = [name+"_%d" %i for i in range(1, repeat+1)]
        for sc in scopes:
            channel  = out.get_shape().as_list()[-1]
            weights_shape = [ks, ks, channel, num_filters]
            biases_shape = [num_filters]
            weights, biases = init_variable(sc, weights_shape, biases_shape, ks*ks*num_filters, pretrained, trainable)
            with tf.variable_scope(sc) as scope:
                out = tf.nn.conv2d(out, weights, strides=[1, strides, strides, 1], padding= padding)
                out = tf.nn.bias_add(out, biases)
                if relu:
                    out = tf.nn.relu(out, name = sc+'_relu')
                if dropout:
                    out = tf.nn.dropout(out, drop_prob)
        return out

    def batch_norm(self, x, n_out): # for mlp and conv layers
        pass

class VGG16net(network):

    """
        VGG16 network
        VGG16 Pretrained model file:https://www.cs.toronto.edu/~frossard/post/vgg16/
    """
    def __init__(self, x,y, numofclass, sess=None, weights_path=None, log_dir=None, verbose=False):
        super(VGG16net, self).__init__(x, y, numofclass, sess,weights_path,  log_dir, verbose)
        if wegiths_path is None:
            weights_path = self.get_model('https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz')
        self.get_feature()
        sess.run(tf.global_variables_initializer())
        self.load_weights(sess)
        self.checkpoint(sess)

    def get_feature(self):
        if self.verbose : print("Creating model graph..", end='')
        """
            VGG16net, pretrained model.
        """
        x = self.X
        # conv1 : 64ch, st 1, pad 1, repeat 2
        conv1 = self.conv(x, 64, name='conv1', repeat=2,relu=True)
        # pool1 : maxpool2, st 2
        pool1 = self._max_pool2_(conv1, "pool1")
        # conv2 : 128ch, repeat 2
        conv2 = self.conv(pool1, 128, name='conv2', repeat=2, relu=True)
        # pool2 : maxpool2, st 2
        pool2 = self._max_pool2_(conv2, "pool2")
        # conv3 : 256ch, repeat 2
        conv3 = self.conv(pool2, 256, name='conv3', repeat=3, relu=True)
        # pool3 : maxpool2, st 2
        pool3 = self._max_pool2_(conv3, "pool3")
        # conv4 : 512ch, repeat 3
        conv4 = self.conv(pool3, 512, name='conv4', repeat=3, relu=True)
        # pool4 : maxpool2, st 2
        pool4 = self._max_pool2_(conv4, "pool4")
        # conv5 : 512ch, repeat 3
        conv5 = self.conv(pool4, 512, name='conv5', repeat=3, relu=True)
        # pool5 : maxpool2, st 2
        pool5 = self._max_pool2_(conv5, "pool5")
        # fc6 : 4096
        fc1 = self.fc(pool5, 4096, 'fc1', relu=True, dropout=True)
        # fc7 : 4096
        fc2 = self.fc(fc6, 4096, 'fc2', relu=True, dropout=True)
        # fc8 : depending on # of classes of the given dataset.
        fc3 = self.fc(fc7, self.numofclass, 'fc3', relu=False, dropout=False,trainable=True, pretrained=False)
        return fc3
