import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19
from modules.resizing import resize_batch

def PixelLossDown(criterion='l1', lr_size=(24, 24)):
    """pixel loss"""
    downsize=tf.keras.layers.experimental.preprocessing.Resizing(lr_size[0], lr_size[1], interpolation='bilinear')
    def _PixelLossDown(sr, hr):
        sr_down = downsize(sr)
        hr_down = downsize(hr)
        if criterion == 'l1':
            return tf.keras.losses.MeanAbsoluteError()(sr_down, hr_down)
        elif criterion == 'l2':
            return tf.keras.losses.MeanSquaredError()(sr_down, hr_down)
    return _PixelLossDown
    
def PixelLoss(criterion='l1'):
    """pixel loss"""
    if criterion == 'l1':
        return tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        return tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))

def gradient_penalty(discriminator, real_images, fake_images, lr=None, refgan=True):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.normal([real_images.shape[0], 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        if refgan:
            pred = discriminator([interpolated, lr], training=True)
        else: 
            pred = discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def ContentLoss(criterion='l1', output_layer=54, before_act=True):
    """content loss"""
    if criterion == 'l1':
        loss_func = tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        loss_func = tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)

    if output_layer == 22:  # Low level feature
        pick_layer = 5
    elif output_layer == 54:  # Hight level feature
        pick_layer = 20
    else:
        raise NotImplementedError(
            'VGG output layer {} is not recognized.'.format(criterion))

    if before_act:
        vgg.layers[pick_layer].activation = None

    fea_extrator = tf.keras.Model(vgg.input, vgg.layers[pick_layer].output)

    @tf.function
    def content_loss(hr, sr):
        # the input scale range is [0, 1] (vgg is [0, 255]).
        # 12.75 is rescale factor for vgg featuremaps.
        preprocess_sr = preprocess_input(sr * 255.) / 12.75
        preprocess_hr = preprocess_input(hr * 255.) / 12.75
        sr_features = fea_extrator(preprocess_sr)
        hr_features = fea_extrator(preprocess_hr)

        return loss_func(hr_features, sr_features)

    return content_loss

def DiscriminatorLoss(gan_type='ragan'):
    """discriminator loss"""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def discriminator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(tf.ones_like(hr), sigma(hr - tf.reduce_mean(sr))) +
            cross_entropy(tf.zeros_like(sr), sigma(sr - tf.reduce_mean(hr))))

    def discriminator_loss(hr, sr):
        real_loss = cross_entropy(tf.ones_like(hr), sigma(hr))
        fake_loss = cross_entropy(tf.zeros_like(sr), sigma(sr))
        return real_loss + fake_loss

    def discriminator_loss_wgan(hr, sr):
        real_loss = tf.reduce_mean(hr)
        fake_loss = tf.reduce_mean(sr)
        return fake_loss - real_loss

    if gan_type == 'ragan':
        return discriminator_loss_ragan
    elif gan_type == 'gan':
        return discriminator_loss
    elif gan_type == 'wgan-gp':
        return discriminator_loss_wgan
    else:
        raise NotImplementedError(
            'Discriminator loss type {} is not recognized.'.format(gan_type))


def GeneratorLoss(gan_type='ragan'):
    """generator loss"""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def generator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(tf.ones_like(sr), sigma(sr - tf.reduce_mean(hr))) +
            cross_entropy(tf.zeros_like(hr), sigma(hr - tf.reduce_mean(sr))))

    def generator_loss(hr, sr):
        return cross_entropy(tf.ones_like(sr), sigma(sr))

    def generator_loss_wgan(hr, sr):
        return -tf.reduce_mean(sr)

    if gan_type == 'ragan':
        return generator_loss_ragan
    elif gan_type == 'gan':
        return generator_loss
    elif gan_type == 'wgan-gp':
        return generator_loss_wgan
    else:
        raise NotImplementedError(
            'Generator loss type {} is not recognized.'.format(gan_type))
