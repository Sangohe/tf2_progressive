import time
import os
import numpy as np
import tensorflow as tf
from tabulate import tabulate
import matplotlib.pyplot as plt

import config

def print_as_table(properties_list, headers=None, tablefmt = 'psql'):
    """Takes a list containing paired values of properties and values
    
    Arguments:
        properties_list {list} -- List filled with lists of len 2 with
        property and value and position zero and one respectively
    
    Keyword Arguments:
        headers {list} -- list of Strings to put as column titles
        (default: {None})
        tablefmt {str} -- string to change how the table will be displayed 
        Some of the possible values are: github, grid, psql, pipe, orgtbl 
        (default: {'psql'})
    """
    if headers:
        print(tabulate(properties_list, headers=headers, tablefmt=tablefmt)) 
    else:
        print(tabulate(properties_list, tablefmt=tablefmt))

def print_log(num_batches_per_epoch, global_epoch, step, global_step, start_time, disc_loss, gen_loss):
    """Writes a summary of the total losses at the current train step
    
    Arguments:
        global_epoch {int} -- Global epoch
        step {int} -- Current step
        global_step {int} -- Global step
        start_time {time} -- Time at the beginning of execution
        disc_loss {float} -- Discriminator's loss at step
        gen_loss {float} -- Generator's loss at step
    """

    epochs = global_epoch + (step+1) / float(num_batches_per_epoch)
    duration = time.time() - start_time
    examples_per_sec = config.batch_size / float(duration)
    print("Epochs: {:.2f} global_step: {} loss_D: {:.3g} loss_G: {:.3g} ({:.2f} examples/sec; {:.3f} sec/batch)".format(
            epochs, global_step, disc_loss, gen_loss, examples_per_sec, duration))        

def print_samples(generator, current_resolution, random_vector_for_sampling=None):
    """Generates fake images using the Generator network and save them to results folder
    
    Arguments:
        current_resolution {[type]} -- [description]
    
    Keyword Arguments:
        random_vector_for_sampling {[type]} -- [description] (default: {None})
    """
    if random_vector_for_sampling is None:
        random_vector_for_sampling = tf.random.uniform([config.num_examples_to_generate, 1, 1, config.noise_dim],
                                                    minval=-1.0, maxval=1.0)
    sample_images = generator(random_vector_for_sampling, current_resolution, 'training')
    print_or_save_sample_images(sample_images.numpy(), config.num_examples_to_generate, is_save=True)            

def print_or_save_sample_images(sample_images, max_print_size=config.num_examples_to_generate,
                                is_square=False, is_save=False, epoch=None,
                                checkpoint_dir=config.checkpoint_dir):
    available_print_size = list(range(1, 26))
    assert max_print_size in available_print_size
    if len(sample_images.shape) == 2:
        size = int(np.sqrt(sample_images.shape[1]))
        channel = 1
    elif len(sample_images.shape) > 2:
        size = sample_images.shape[1]
        channel = sample_images.shape[3]
    else:
        ValueError('Not valid a shape of sample_images')
  
    if not is_square:
        print_images = sample_images[:max_print_size, ...]
        print_images = print_images.reshape([max_print_size, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, max_print_size * size, channel])
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)

        fig = plt.figure(figsize=(max_print_size, 1))
        plt.imshow(print_images * 0.5 + 0.5)#, cmap='gray')
        plt.axis('off')
    
    else:
        num_columns = int(np.sqrt(max_print_size))
        max_print_size = int(num_columns**2)
        print_images = sample_images[:max_print_size, ...]
        print_images = print_images.reshape([max_print_size, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, max_print_size * size, channel])
        print_images = [print_images[:,i*size*num_columns:(i+1)*size*num_columns] for i in range(num_columns)]
        print_images = np.concatenate(tuple(print_images), axis=0)
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)
        
        fig = plt.figure(figsize=(num_columns, num_columns))
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        # plt.imshow(print_images * 0.5 + 0.5)#, cmap='gray')
        plt.axis('off')
        
    if is_save and epoch is not None:
        filepath = os.path.join(checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.savefig(filepath)
    else:
        plt.show()

def save_image_grid(generator, current_resolution, checkpoint_dir, global_step, random_vector_for_sampling=None):
    """[summary]
    
    Arguments:
        generator {tf.keras.Model} -- The generator network
        checkpoint_dir {String} -- The checkpoint directory path to save images
        global_step {int} -- Execution's global step
    
    Keyword Arguments:
        random_vector_for_sampling {Tensor} -- Seed (default: {None})
    """
    if random_vector_for_sampling is None:
        random_vector_for_sampling = tf.random.uniform([config.num_examples_to_generate, 1, 1, config.noise_dim],
                                                    minval=-1.0, maxval=1.0)
    images = generator(random_vector_for_sampling, current_resolution, 'training').numpy()
    np.save(os.path.join(checkpoint_dir, 'fake{:06}'.format(global_step)), images)
    print('image range [{},{}]'.format(images.min(), images.max()))

    # num, height, width, _ = images.shape
    # assert height == width
    # dim = np.sqrt(num).astype(int)
    # rows, row_num = [], 0
    
    # while row_num < dim:
    #     rows.append(np.hstack(images[row_num * dim:(row_num + 1) * dim]))
    #     row_num+=1
        
    # np.save(os.path.join(checkpoint_dir, 'fake{:06}'.format(global_step)), np.vstack(rows))
    

def augment(data):
    #corre un pixel a lo largo de x y los concatena.
    off_x_p = data.copy()    
    off_x_p[:,1:,:,:] = off_x_p[:,:-1,:,:]
    off_x_m = data.copy()
    off_x_p[:,:-1,:,:] = off_x_p[:,1:,:,:]
    data = np.concatenate((data,off_x_p,off_x_m), axis = 0)
    #corre un pixel a lo largo de y y los concatena.
    off_y_p = data.copy()
    off_y_p[:,:,1:,:] = off_y_p[:,:,:-1,:]
    off_y_m = data.copy()
    off_y_m[:,:,:-1,:] = off_y_m[:,:,1:,:]
    data = np.concatenate((data,off_y_p,off_y_m), axis = 0)    
    return data