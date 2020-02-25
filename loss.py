import tensorflow as tf

def WGANLoss(logits, is_real=True):
    """Computes Wasserstain GAN loss

    Args:
        logits (`2-rank Tensor`): logits
        is_real (`bool`): boolean, Treu means `-` sign, False means `+` sign.

    Returns:
        loss (`0-rank Tensor`): the WGAN loss value.
    """
    if is_real:
        return -tf.reduce_mean(logits)
    else:
        return tf.reduce_mean(logits)

def GANLoss(logits, is_real=True, use_lsgan=True):
    """Computes standard GAN or LSGAN loss between `logits` and `labels`.

    Args:
        logits (`2-rank Tensor`): logits.
        is_real (`bool`): True means `1` labeling, False means `0` labeling.
        use_lsgan (`bool`): True means LSGAN loss, False means standard GAN loss.

    Returns:
        loss (`0-rank Tensor`): the standard GAN or LSGAN loss value. (binary_cross_entropy or mean_squared_error)
    """
    # Define losses
    bce = tf.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.losses.MeanSquaredError()

    if is_real:
        labels = tf.ones_like(logits)
    else:
        labels = tf.zeros_like(logits)
        
    if use_lsgan:
        loss = mse(labels, tf.nn.sigmoid(logits))
    else:
        loss = bce(labels, logits)
        
    return loss        

def discriminator_loss(real_logits, fake_logits):
    # losses of real with label "1"
    real_loss = WGANLoss(logits=real_logits, is_real=True)
    # losses of fake with label "0"
    fake_loss = WGANLoss(logits=fake_logits, is_real=False)
    
    return real_loss + fake_loss    

def generator_loss(fake_logits):
    # losses of Generator with label "1" that used to fool the Discriminator
    return WGANLoss(logits=fake_logits, is_real=True)    