import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import misc
import config

def get_data(dataset_name):
    """Returns the set of images and labels from a desired dataset.
    
    Arguments:
        dataset_name {String} -- The name of the dataset to retrieve
        images and labels
    
    Returns:
        tuple -- Returns the normalized images with their respective labels
    """
    if dataset_name == 'cifar10':
        (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
    if dataset_name == 'cifar10-planes':
        (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
        indx = np.where(train_labels == 0)[0] # we choose only one specific domain from the dataset  0 -> Avioncitos
        #train_images=misc.augment(train_images[indx])
        train_images=train_images[indx]
        train_labels=np.ones((train_images.shape))
    if(train_images.shape[0]%config.batch_size!=0): # we reduce the dataset to be taken by batches.  
        train_images=train_images[:-(train_images.shape[0]%config.batch_size),...]
    # Normalize the images to [-1, 1]
    train_images = train_images.astype('float32')
    train_images = (train_images - 127.5) / 127.5

    return (train_images, train_labels)

def normalize(images):
    """Takes a set of images and normalize its values to [-1, 1]
    
    Arguments:
        images {Array} -- Set of images
    
    Returns:
        Array -- [-1, 1] normalized images
    """
    images = images.astype('float32')
    images = (images - 127.5) / 127.5
    return images

def resize(image, size):
    """Converts an image into a tensor and resize it to [size, size]
    
    Arguments:
        image {tensor} -- Image to resize
        size {int} -- Desired size for image
    
    Returns:
        [type] -- Tensor containing a resized image
    """
    return tf.image.resize(image, [size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def create_dataset(images, resolution, batch_size, normalize=True):
    """Takes an array of images and creates a Tensorflow dataset
    with dimensions: [resolution, resolution]
    
    Arguments:
        images {array} -- An array containing all the images
        resolution {int} -- Final resolution for each image
        batch_size {int} -- Size to make batches from tensor
    
    Keyword Arguments:
        normalize {boolean} -- Flag (default: {True})

    Returns:
        [type] -- Tensorflow dataset object to feed the networks
    """
    if normalize:
        images = images.astype('float32')
        images = (images - 127.5) / 127.5

    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=images.shape[0])
    dataset = dataset.map(lambda img: resize(img, resolution))
    dataset = dataset.batch(batch_size=batch_size)
    
    return dataset