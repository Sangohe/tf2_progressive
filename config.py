import os

desc                                = 'pgan'                    # Description string included in result subdir name.
random_seed                         = 69                        # Global random seed.
batch_size                          = 16                        # Size for batches
noise_dim                           = 512                       # Size for the latent space vector
learning_rate_D                     = 1e-3                      # Discriminator's learning rate
learning_rate_G                     = 1e-3                      # Discriminator's learning rate
gp_lambda                           = 10                        # Grandient Penalty Lambda
num_examples_to_generate            = 16                        # Number of fake samples to generate
training_phase_epoch                = 1                         # Training phase epoch
transition_phase_epoch              = 1                         # Transition phase epoch
print_steps                         = 20                        # Step to print log results and generate images
save_images_epochs                  = 1                         # Save images at epoch number
save_model_epochs                   = 1                         # Save model at epoch

# Dataset (choose one)
#desc += '-cifar10';             dataset_name = 'cifar10';           resolution = 32;
desc += '-cifar10-planes';             dataset_name = 'cifar10-planes';           resolution = 32;
# desc += '-acdc';                dataset_name = 'acdc';              resolution = 256;    

checkpoint_dir = os.path.join('results', desc)