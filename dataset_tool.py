import os
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Create a directory from a numpy file or a set of images')
parser.add_argument('--images_path', help='Path where the images are located')
parser.add_argument('--labels_path', help='Path where the labels are located')
parser.add_argument('--dataset_name', help='The name of the dataset you which to create')
parser.add_argument('--target_res', help='Target resolution for the set of images', type=int)
parser.add_argument('--numpy', help='If True the source file is a npy file', action='store_true')
args = parser.parse_args()

def create_dataset_dir(images_path, dataset_name, target_res, labels_path=None, numpy=False):

    print("Creating dataset")
    dir_name = os.path.join('datasets', dataset_name)
    print(dir_name)
    lods = [lod for lod in range(2, int(np.log2(target_res)) + 1)]

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('Dataset directory created: {}'.format(dir_name))
    else:
        raise ValueError('Directory already exists.')

    if numpy:
        numpy_images = np.load(images_path)
        for lod in lods:
            print('')
            res_images = np.asarray([cv2.resize(img, (2**lod, 2**lod)) for img in numpy_images])
            np.save(os.path.join(dir_name, '{}-images-r{:02}.npy'.format(dataset_name, lod)), res_images)
            print('Images of resolution {} saved to: {}'.format(2**lod, os.path.join(dir_name, '{}-images-r{:02}.npy'.format(dataset_name, lod))))
        if labels_path:
            numpy_labels = np.load(labels_path)
            np.save(os.path.join(dir_name, '{}-labels-rxx.npy'.format(dataset_name)), numpy_labels)
            print('Labels saved to: {}'.format(os.path.join(dir_name, '{}-labels-rxx.npy'.format(dataset_name))))
    else:
        pass

if args.dataset_name:
    create_dataset_dir(args.images_path, args.dataset_name, args.target_res, labels_path=args.labels_path, numpy=args.numpy)