import numpy as np
import matplotlib.pyplot as plt
import pickle

CIFAR10_LABELS_LIST = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
def view_cifar100(file, spec = False, no = None):
    f = open(file,'rb')
    datadict = pickle.load(f, encoding='bytes')
    f.close()
    X = datadict[b'data']
    Y = datadict[b'fine_labels']
    X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)
    labels = []
    if not spec:
        fig, axes1 = plt.subplots(5,5,figsize=(3,3))
        for i in range(5):
            for k in range(5):
                j = np.random.choice(range(len(X)))
                axes1[i][k].set_axis_off()
                axes1[i][k].imshow(X[j:j+1][0])
                labels.append(CIFAR100_LABELS_LIST[Y[j]])
        for i in range(5):
            print(labels[i*5:i*5+5])
    elif spec & no <= 50000:
        fig = plt.imshow(X[no:no+1][0])
        print(CIFAR100_LABELS_LIST[Y[no]])

def view_cifar10(file, spec = False, no = None):
    f = open(file,'rb')
    datadict = pickle.load(f, encoding='bytes')
    f.close()
    X = datadict[b'data']
    Y = datadict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)
    labels = []
    if not spec:
        fig, axes1 = plt.subplots(5,5,figsize=(3,3))
        for i in range(5):
            for k in range(5):
                j = np.random.choice(range(len(X)))
                axes1[i][k].set_axis_off()
                axes1[i][k].imshow(X[j:j+1][0])
                labels.append(CIFAR10_LABELS_LIST[Y[j]])
        for i in range(5):
            print(labels[i*5:i*5+5])
    elif spec & no <= 10000:
        fig = plt.imshow(X[no:no+1][0])
        print(CIFAR10_LABELS_LIST[Y[no]])
