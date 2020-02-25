import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  

# Libraries
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Python files
import misc
import loss
import config
import dataset
import networks

# Helper functions
def get_discriminator_tvars(current_resolution):
    d_tvars = []
    for var in discriminator.trainable_variables:
        if current_resolution >= int(var.name.split('/')[1].split('x')[0]):
            d_tvars.append(var)
        
    return d_tvars

def get_generator_tvars(current_resolution):
    g_tvars = []
    for var in generator.trainable_variables:
        if current_resolution >= int(var.name.split('/')[1].split('x')[0]):
            g_tvars.append(var)
    
    return g_tvars    

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def discriminator_train_step(images, current_resolution, current_phase, alpha=0.0):
    # generating noise from a uniform distribution
    noise = tf.random.uniform([config.batch_size, config.noise_dim], minval=-1.0, maxval=1.0)

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, current_resolution, current_phase, alpha)

        real_logits = discriminator(images, current_resolution, current_phase, alpha)
        fake_logits = discriminator(generated_images, current_resolution, current_phase, alpha)
    
        # interpolation of x hat for gradient penalty : epsilon * real image + (1 - epsilon) * generated image
        epsilon = tf.random.uniform([config.batch_size, 1, 1, 1])
        # epsilon = tf.expand_dims(tf.stack([tf.stack([epsilon]*current_resolution, axis=1)]*current_resolution, axis=1), axis=3)
        interpolated_images_4gp = epsilon * images + (1. - epsilon) * generated_images
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images_4gp)
            interpolated_images_logits = discriminator(interpolated_images_4gp, current_resolution, current_phase, alpha)
        
        gradients_of_interpolated_images = gp_tape.gradient(interpolated_images_logits, interpolated_images_4gp)
        norm_grads = tf.sqrt(tf.reduce_sum(tf.square(gradients_of_interpolated_images), axis=[1, 2, 3]))
        gradient_penalty_loss = tf.reduce_mean(tf.square(norm_grads - 1.))
        
        disc_loss = loss.discriminator_loss(real_logits, fake_logits) + config.gp_lambda * gradient_penalty_loss
        gen_loss = loss.generator_loss(fake_logits)

    d_tvars = get_discriminator_tvars(current_resolution)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d_tvars)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d_tvars))
        
    return gen_loss, disc_loss

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def generator_train_step(current_resolution, current_phase, alpha=0.0):
    # generating noise from a uniform distribution
    noise = tf.random.uniform([config.batch_size, config.noise_dim], minval=-1.0, maxval=1.0)

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, current_resolution, current_phase, alpha)

        fake_logits = discriminator(generated_images, current_resolution, current_phase, alpha)
        gen_loss = loss.generator_loss(fake_logits)

    g_tvars = get_generator_tvars(current_resolution)
    gradients_of_generator = gen_tape.gradient(gen_loss, g_tvars)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g_tvars))

# Load the dataset
(train_images, train_labels) = dataset.get_data(config.dataset_name)
train_datasets = [dataset.create_dataset(train_images, 2**lod, config.batch_size) for lod in range(2, np.log2(config.resolution).astype(int) + 1)]

# Train parameters
dataset_attributes = [
    ['Dataset name', config.dataset_name], 
    ['Number of images in dataset', train_images.shape[0]], 
    ['Images values are in', '[{}, {}]'.format(np.min(train_images), np.max(train_images))]
]

misc.print_as_table(dataset_attributes, headers=['Parameter', 'Value'])

# Build the networks
generator = networks.Generator()
discriminator = networks.Discriminator()

# Optimizers
discriminator_optimizer = tf.keras.optimizers.Adam(config.learning_rate_D, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
generator_optimizer = tf.keras.optimizers.Adam(config.learning_rate_G, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

# Create directory if it doesn't exist to train
if not tf.io.gfile.exists(config.checkpoint_dir):
    tf.io.gfile.makedirs(config.checkpoint_dir)
checkpoint_prefix = os.path.join(config.checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Use a constant latent space vector for generation to watch for improvements during training
const_random_vector_for_saving = tf.random.uniform([config.num_examples_to_generate, config.noise_dim], minval=-1.0, maxval=1.0, seed=config.random_seed)

# Initialize full size networks for making full size static graph
TARGET_SIZE = config.resolution
_, _ = discriminator_train_step(tf.random.normal([config.batch_size, TARGET_SIZE, TARGET_SIZE, 3]), TARGET_SIZE, 'transition')
generator_train_step(TARGET_SIZE, 'transition')

generator.summary()
discriminator.summary()

print('Start Training.')
num_batches_per_epoch = int(train_images.shape[0] / config.batch_size)
global_step = 1 
global_epoch = 0
num_learning_critic = 0

# 4 x 4 training phase
current_resolution = 4
for epoch in range(config.training_phase_epoch):
    for step, images in enumerate(train_datasets[0]):
        start_time = time.time()
        
        gen_loss, disc_loss = discriminator_train_step(images, current_resolution, 'training')
        generator_train_step(current_resolution, 'training')
        if global_step % (config.print_steps//current_resolution) == 0:
            misc.print_log(num_batches_per_epoch, global_epoch, step, global_step, start_time, disc_loss, gen_loss)
            misc.print_samples(generator, current_resolution, const_random_vector_for_saving)
        
        global_step += 1
    global_epoch += 1


for resolution, train_dataset in enumerate(train_datasets[1:]):
    current_resolution = 2**(resolution+3)
    
    # transition phase
    for epoch in range(config.transition_phase_epoch):
        for step, images in enumerate(train_dataset):
            start_time = time.time()
            alpha = (epoch * num_batches_per_epoch + step) / float(config.transition_phase_epoch * num_batches_per_epoch)
            gen_loss, disc_loss = discriminator_train_step(images, current_resolution, 'transition', alpha)
            generator_train_step(current_resolution, 'transition', alpha)
        
            if global_step % (config.print_steps//current_resolution) == 0:
                misc.print_log(num_batches_per_epoch, global_epoch, step, global_step, start_time, disc_loss, gen_loss)
                misc.print_samples(generator, current_resolution, const_random_vector_for_saving)
        
            global_step += 1
        global_epoch += 1
        
    # training phase
    for epoch in range(config.training_phase_epoch):
        for step, images in enumerate(train_dataset):
            start_time = time.time()
            gen_loss, disc_loss = discriminator_train_step(images, current_resolution, 'training')
            generator_train_step(current_resolution, 'training')
        
            if global_step % (config.print_steps//current_resolution) == 0:
                misc.print_log(num_batches_per_epoch, global_epoch, step, global_step, start_time, disc_loss, gen_loss)
                misc.print_samples(generator, current_resolution, const_random_vector_for_saving)
        
            global_step += 1
        global_epoch += 1


        if (epoch + 1) % config.save_images_epochs == 0:
            print("This images are saved at {} epoch".format(epoch+1))
            sample_images = generator(const_random_vector_for_saving, training=False)
            misc.print_or_save_sample_images(sample_images.numpy(), config.num_examples_to_generate,
                                        is_square=True, is_save=True, epoch=epoch+1,
                                        checkpoint_dir=config.checkpoint_dir)

        # saving (checkpoint) the model every save_epochs
        if (epoch + 1) % config.save_model_epochs == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
    
print('Training Done.')