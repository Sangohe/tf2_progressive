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
  def __init__(self, TARGET_SIZE):
    super(Generator, self).__init__()
    self.block1 = G_Init_Block(512, '4x4')      # [bs, 4, 4, 512]
    self.block2 = G_Block(512, '8x8')           # [bs, 8, 8, 512]
    self.block3 = G_Block(512, '16x16')         # [bs, 16, 16, 512]
    self.block4 = G_Block(512, '32x32')         # [bs, 32, 32, 512]
    self.to_RGB = to_RGB('0xto_rgb')            # [bs, height, width, 3]
    self.upsample = layers.UpSampling2D()       # Double each layer

    if TARGET_SIZE > 32:
        self.block5 = G_Block(512, '64x64')     # [bs, 64, 64, 512]
    if TARGET_SIZE > 64:       
        self.block6 = G_Block(512, '128x128')   # [bs, 128, 128, 512]
    if TARGET_SIZE > 128:
        self.block7 = G_Block(512, '256x256')   # [bs, 256, 256, 512]
    if TARGET_SIZE > 256:
        self.block8 = G_Block(512, '512x512')   # [bs, 512, 512, 512]
    if TARGET_SIZE > 512:
        self.block9 = G_Block(512, '1024x1024') # [bs, 1024, 1024, 512]

  def call(self, inputs, current_resolution, current_phase, alpha=0.0):
    """Run the model."""
    assert current_resolution in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    assert current_phase in ['training', 'transition']

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

    if current_resolution > 32:
        outputs = block5 = self.block5(outputs)
        prev_outputs = block4        
    
    if current_resolution > 64:
        outputs = block6 = self.block6(outputs)
        prev_outputs = block5

    if current_resolution > 128:
        outputs = block7 = self.block7(outputs)
        prev_outputs = block6

    if current_resolution > 256:
        outputs = block8 = self.block8(outputs)
        prev_outputs = block7

    if current_resolution > 512:
        outputs = block9 = self.block9(outputs)
        prev_outputs = block8

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
    def __init__(self, TARGET_SIZE):
        super(Discriminator, self).__init__()
        self.from_RGB = from_RGB(512, '0xfrom_rgb')     # [bs, height, width, 3]
        if TARGET_SIZE > 512:
            self.block1 = D_Block(512, 512, '1024x1024')    # [bs, 512, 512, 512]
        if TARGET_SIZE > 256:
            self.block2 = D_Block(512, 512, '512x512')      # [bs, 256, 256, 512]
        if TARGET_SIZE > 128:
            self.block3 = D_Block(512, 512, '256x256')      # [bs, 128, 128, 512]
        if TARGET_SIZE > 64:
            self.block4 = D_Block(512, 512, '128x128')      # [bs, 64, 64, 512]
        if TARGET_SIZE > 32:
            self.block5 = D_Block(512, 512, '64x64')        # [bs, 32, 32, 512]
        self.block6 = D_Block(512, 512, '32x32')        # [bs, 16, 16, 512]
        self.block7 = D_Block(512, 512, '16x16')        # [bs, 8, 8, 512]
        self.block8 = D_Block(512, 512, '8x8')          # [bs, 4, 4, 512]
        self.block9 = D_Last_Block(512, 512, '4x4')     # [bs, 1]
        self.downsample = layers.AveragePooling2D()

    def call(self, inputs, current_resolution, current_phase, alpha=0.0):
        """Run the model."""
        assert current_resolution in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        assert current_phase in ['training', 'transition']
        
        new_inputs = self.from_RGB(inputs)
        
        if current_phase == 'transition':
            smoothing_inputs = self.from_RGB(self.downsample(inputs))
        
        if current_resolution > 512:
            new_inputs = block1 = self.block1(new_inputs)
        if current_phase == 'transition' and current_resolution == 1024:
            new_inputs = alpha * block1 + (1. - alpha) * smoothing_inputs

        if current_resolution > 256:
            new_inputs = block2 = self.block2(new_inputs)
        if current_phase == 'transition' and current_resolution == 512:
            new_inputs = alpha * block2 + (1. - alpha) * smoothing_inputs

        if current_resolution > 128:
            new_inputs = block3 = self.block3(new_inputs)
        if current_phase == 'transition' and current_resolution == 256:
            new_inputs = alpha * block3 + (1. - alpha) * smoothing_inputs

        if current_resolution > 64:
            new_inputs = block4 = self.block4(new_inputs)
        if current_phase == 'transition' and current_resolution == 128:
            new_inputs = alpha * block4 + (1. - alpha) * smoothing_inputs

        if current_resolution > 32:
            new_inputs = block5 = self.block5(new_inputs)
        if current_phase == 'transition' and current_resolution == 64:
            new_inputs = alpha * block5 + (1. - alpha) * smoothing_inputs
        
        if current_resolution > 16:
            new_inputs = block6 = self.block6(new_inputs)
        if current_phase == 'transition' and current_resolution == 32:
            new_inputs = alpha * block6 + (1. - alpha) * smoothing_inputs
        
        if current_resolution > 8:
            new_inputs = block7 = self.block7(new_inputs)
        if current_phase == 'transition' and current_resolution == 16:
            new_inputs = alpha * block7 + (1. - alpha) * smoothing_inputs
        
        if current_resolution > 4:
            new_inputs = block8 = self.block8(new_inputs)
        if current_phase == 'transition' and current_resolution == 8:
            new_inputs = alpha * block8 + (1. - alpha) * smoothing_inputs
        
        discriminator_logits = self.block9(new_inputs)
        
        return discriminator_logits        