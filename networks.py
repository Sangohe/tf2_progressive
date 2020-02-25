# Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class PixelNormalization(tf.keras.Model):
    def __init__(self, epsilon=1e-8, name='PixelNorm'):
        super(PixelNormalization, self).__init__(name=name)
        self.epsilon = epsilon
    
    def call(self, inputs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)

class G_Block(tf.keras.Model):
    def __init__(self, filters, name):
        super(G_Block, self).__init__(name=name)
        self.upsample = layers.UpSampling2D()
        self.conv1 = layers.Conv2D(filters, 3, padding='same', activation=tf.nn.leaky_relu,
                               kernel_initializer='he_normal')
        self.conv2 = layers.Conv2D(filters, 3, padding='same', activation=tf.nn.leaky_relu,
                                kernel_initializer='he_normal')
        self.pn = PixelNormalization()
    
    def call(self, inputs):
        up = self.upsample(inputs)
        conv1 = self.conv1(up)
        conv1 = self.pn(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.pn(conv2)
        
        return conv2

class G_Init_Block(tf.keras.Model):
    def __init__(self, filters, name):
        super(G_Init_Block, self).__init__(name=name)
        self.filters = filters
        self.dense = layers.Dense(filters * 4 * 4, activation=tf.nn.leaky_relu,
                                kernel_initializer='he_normal')
        self.conv = layers.Conv2D(filters, 3, padding='same', activation=tf.nn.leaky_relu,
                                kernel_initializer='he_normal')
        self.pn = PixelNormalization()
        
    def call(self, inputs):
        dense = self.dense(inputs)
        dense = self.pn(dense)
        dense = tf.reshape(dense, shape=[-1, 4, 4, self.filters])
        conv = self.conv(dense)
        conv = self.pn(conv)
        
        return conv

class to_RGB(tf.keras.Model):
    def __init__(self, name):
        super(to_RGB, self).__init__(name=name)
        self.conv = layers.Conv2D(3, 1, padding='same', kernel_initializer='he_normal')
        
    def call(self, inputs):
        conv = self.conv(inputs)
        
        return conv

class Generator(tf.keras.Model):
  """Build a generator that maps latent space to images: G(z): z -> x
  """
  def __init__(self):
    super(Generator, self).__init__()
    self.block1 = G_Init_Block(512, '4x4')  # [bs, 4, 4, 512]
    self.block2 = G_Block(512, '8x8')       # [bs, 8, 8, 512]
    self.block3 = G_Block(512, '16x16')     # [bs, 16, 16, 512]
    self.block4 = G_Block(512, '32x32')     # [bs, 32, 32, 512]
    self.to_RGB = to_RGB('0xto_rgb')        # [bs, height, width, 3]
    self.upsample = layers.UpSampling2D()

  def call(self, inputs, current_resolution, current_phase, alpha=0.0):
    """Run the model."""
    #assert current_resolution in [4, 8, 16, 32]
    #assert current_phase in ['training', 'transition']

    # inputs: [1, 1, 512]
    outputs = block1 = self.block1(inputs)

    if current_resolution > 4:
        outputs = block2 = self.block2(outputs)
        prev_outputs = block1
        
    if current_resolution > 8:
        outputs = block3 = self.block3(outputs)
        prev_outputs = block2
        
    if current_resolution > 16:
        outputs = block4 = self.block4(outputs)
        prev_outputs = block3

    generated_images = self.to_RGB(outputs)

    if current_phase == 'transition':
        prev_outputs = self.upsample(self.to_RGB(prev_outputs))
        generated_images = alpha * generated_images + (1. - alpha) * prev_outputs

    return generated_images


class D_Block(tf.keras.Model):
    def __init__(self, filters1, filters2, name):
        super(D_Block, self).__init__(name=name)
        self.conv1 = layers.Conv2D(filters1, 3, padding='same', activation=tf.nn.leaky_relu,
                                kernel_initializer='he_normal')
        self.conv2 = layers.Conv2D(filters2, 3, padding='same', activation=tf.nn.leaky_relu,
                                kernel_initializer='he_normal')
        self.downsample = layers.AveragePooling2D()
        
    def call(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        downsample = self.downsample(conv2)
        
        return downsample

class D_Last_Block(tf.keras.Model):
    def __init__(self, filters1, filters2, name):
        super(D_Last_Block, self).__init__(name=name)
        self.conv1 = layers.Conv2D(filters1, 3, padding='same', activation=tf.nn.leaky_relu,
                                kernel_initializer='he_normal')
        self.conv2 = layers.Conv2D(filters1, 4, padding='same', activation=tf.nn.leaky_relu,
                                kernel_initializer='he_normal')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, kernel_initializer='he_normal')
        
    def call(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        flatten = self.flatten(conv2)
        dense = self.dense(flatten)

        return dense    

class from_RGB(tf.keras.Model):
    def __init__(self, filters, name):
        super(from_RGB, self).__init__(name=name)
        self.conv = layers.Conv2D(filters, 1, padding='same', activation=tf.nn.leaky_relu,
                                kernel_initializer='he_normal')
        
    def call(self, inputs):
        conv = self.conv(inputs)
        
        return conv

class Discriminator(tf.keras.Model):
    """Build a discriminator that discriminate real image x whether real or fake.
        D(x): x -> [0, 1]
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.from_RGB = from_RGB(512, '0xfrom_rgb')   # [bs, height, width, 3]
        self.block1 = D_Block(512, 512, '32x32')    # [bs, 16, 16, 32]
        self.block2 = D_Block(512, 512, '16x16')    # [bs, 8, 8, 64]
        self.block3 = D_Block(512, 512, '8x8')      # [bs, 4, 4, 128]
        self.block4 = D_Last_Block(512, 512, '4x4') # [bs, 1]
        self.downsample = layers.AveragePooling2D()

    def call(self, inputs, current_resolution, current_phase, alpha=0.0):
        """Run the model."""
        #assert current_resolution in [4, 8, 16, 32]
        #assert current_phase in ['training', 'transition']
        
        new_inputs = self.from_RGB(inputs)
        
        if current_phase == 'transition':
            smoothing_inputs = self.from_RGB(self.downsample(inputs))
        
        if current_resolution > 16:
            new_inputs = block1 = self.block1(new_inputs)
        if current_phase == 'transition' and current_resolution == 32:
            new_inputs = alpha * block1 + (1. - alpha) * smoothing_inputs
        
        if current_resolution > 8:
            new_inputs = block2 = self.block2(new_inputs)
        if current_phase == 'transition' and current_resolution == 16:
            new_inputs = alpha * block2 + (1. - alpha) * smoothing_inputs
        
        if current_resolution > 4:
            new_inputs = block3 = self.block3(new_inputs)
        if current_phase == 'transition' and current_resolution == 8:
            new_inputs = alpha * block3 + (1. - alpha) * smoothing_inputs
        
        discriminator_logits = self.block4(new_inputs)
        
        return discriminator_logits        