import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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

    # Normalize the images to [-1, 1]
    train_images = train_images.astype('float32')
    train_images = (train_images - 127.5) / 127.5

    return (train_images, train_labels)

def resize(image, size):
    """Converts an image into a tensor and resize it to [size, size]
    
    Arguments:
        image {tensor} -- Image to resize
        size {int} -- Desired size for image
    
    Returns:
        [type] -- Tensor containing a resized image
    """
    return tf.image.resize(image, [size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def create_dataset(images, resolution, batch_size):
    """Takes an array of images and creates a Tensorflow dataset
    with dimensions: [resolution, resolution]
    
    Arguments:
        images {array} -- An array containing all the images
        resolution {int} -- Final resolution for each image
        batch_size {int} -- Size to make batches from tensor
    
    Returns:
        [type] -- Tensorflow dataset object to feed the networks
    """
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=images.shape[0])
    dataset = dataset.map(lambda img: resize(img, resolution))
    dataset = dataset.batch(batch_size=batch_size)
    
    return dataset