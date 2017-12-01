import tensorflow as tf
import numpy as np
from . import model



"""
    BaseLine Progress

    #0 images argumentation from given dataset. -> How to?
    #1 fine tuning network for given dataset.
    #2 from the network, collect activation vector from penultimate layer if the best score is label among each activation.
    #3 compute Mean AV.
    #4 fit weibull distribution.
    #5 Meta-Recognition
    #6 Profit?!

    What i have to determine.

    #1 Selecting Dataset
    #2 argumentation strategy.

"""
def train(models, FLAGS): # #0 ~ #4
    """
        Inputs
        - models : pre-define network architectures.
        - FLAGS : pass by argparse, it includes some kinds of hyper-parameters.
    """
    network = models[FLAGS.net]['network']
    meta_dataset = FLAGS.dataset
    weights_path = models[FLAGS.net]['weights_path']
    verbose = FLAGS.debug
    img_size = network['img_size']
    network_name = network['name']

    imgs = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], img_size[2]], name="input_images")
    labels = tf.placeholder(tf.int32, [None], name="label_images")

    """
    How to construct mini-batch?
     #1 consists of argumented images from one image.
     #2 shuffle all. <- selected
    """
    sess = tf.Session()
    log_dir = "./"+network_name
    net = network['network'](imgs, labels, meta_dataset['numofclass'], sess, weights_path, log_dir, verbose)
    logits = net.get_feature()
    print("Setting up saver..", end='')
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    print("Done")

    loss = tf.nn.sparce_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="softmax"))

    optimizer = tf.train.GradientDescentOptimizer() # Other option should be considered.
    grads = optimizer.compute_gradient(loss, tf.trainable_variable())
    train_op = optimizer.apply_gradients(grads)

    #
    show = 2
    tf.summary.image("INPUT_imgaes", image, max_outputs = show)
    tf.summary.image("GROUND_TRUTH", meta_dataset[tf.cast(labels, tf.uint8)], max_outputs=show)
    tf.summary.image("PREDICTION", meta_dataset[tf.cast(tf.argmax(logits, axis=1))], max_outputs=show)
    tf.summary.scalar("softmax", loss)

    print("Setting up summary..", end='')
    sum_op = tf.summary_merge_all()
    print("Done")

    print("Getting data by reader..", end='')
    train_data_reader = foo #TODO :foo has to be implemented
    print("Done.")

    for i in range(MAX_ITER): #TODO : training
        train_images, train_labels = foo.foo # it has to be implemented
        feed_dict = {imgs : train_images, labels = train_labels}
        sess.run(train_op, feed_dict=feed_dict)
        if i%5 == 0:
            train_loss, summary_str = sess.run([loss, sum_op], fedd_dict=feed_dict)
            print("Step : %d, Train loss : %g" %(i, train_loss))
            summary_writer.add_summary(summary_str, i)

    """
        ALGORITHM #1
    """
    #TODO : COMPUTE MEAN ACTIVATION VECTOR
    #
    # feed_forwarding cropped images to classify, if its best score is equal to label,
    # collect it to use for computing MAV for the class which the image belongs to.
    # Then, compute MAV.
    #

    #TODO : fir weibull distributtion
    #
    # use libMR library to fit weibull function by MAV
    #
def test(models, FLAGS): # #5
    pass

    """
        ALGORITHM #2
    """
    #TODO : Do Meta-Recognition
    #
    # use ALGORITHM #2
    #
