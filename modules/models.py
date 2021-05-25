import functools
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, LeakyReLU
from modules.resizing import imresize_np

def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer with scale."""
    scale = 2. * scale
    return tf.keras.initializers.VarianceScaling(
        scale=scale, mode='fan_in', distribution="truncated_normal", seed=seed)


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ResDenseBlock_5C(tf.keras.layers.Layer):
    """Residual Dense Block"""
    def __init__(self, nf=64, gc=32, res_beta=0.2, wd=0., name='RDB5C',
                 **kwargs):
        super(ResDenseBlock_5C, self).__init__(name=name, **kwargs)
        # gc: growth channel, i.e. intermediate channels
        self.res_beta = res_beta
        lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
        _Conv2DLayer = functools.partial(
            Conv2D, kernel_size=3, padding='same',
            kernel_initializer=_kernel_init(0.1), bias_initializer='zeros',
            kernel_regularizer=_regularizer(wd))
        self.conv1 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv2 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv3 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv4 = _Conv2DLayer(filters=gc, activation=lrelu_f())
        self.conv5 = _Conv2DLayer(filters=nf, activation=lrelu_f())

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(tf.concat([x, x1], 3))
        x3 = self.conv3(tf.concat([x, x1, x2], 3))
        x4 = self.conv4(tf.concat([x, x1, x2, x3], 3))
        x5 = self.conv5(tf.concat([x, x1, x2, x3, x4], 3))
        return x5 * self.res_beta + x



class ResInResDenseBlock(tf.keras.layers.Layer):
    """Residual in Residual Dense Block"""
    def __init__(self, apply_noise=False, nf=64, gc=32, res_beta=0.2, wd=0., name='RRDB',
                 **kwargs):
        super(ResInResDenseBlock, self).__init__(name=name, **kwargs)
        self.apply_noise = apply_noise
        self.res_beta = res_beta
        self.rdb_1 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_2 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.rdb_3 = ResDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.applynoise = ApplyNoise()

    def call(self, x):
        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        out = out * self.res_beta + x
        if self.apply_noise:
            out = self.applynoise(out)
        return out

class ReceptiveFieldBlock(tf.keras.layers.Layer):
    def __init__(self, nf=64, gc=16, wd=0., name='RFB',**kwargs):
        r""" Single RFB block."""
        super(ReceptiveFieldBlock, self).__init__()
        lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
        _Conv2DLayer = functools.partial(
            Conv2D, padding='same',
            kernel_initializer=_kernel_init(0.1), bias_initializer='zeros',
            kernel_regularizer=_regularizer(wd))
        # shortcut layer
        self.shortcut = _Conv2DLayer(filters=nf, kernel_size=1, activation=None)

        self.branch1_1 = _Conv2DLayer(filters=gc, kernel_size=1, activation=lrelu_f())
        self.branch1_2 = _Conv2DLayer(filters=gc, kernel_size=3, activation=None)

        self.branch2_1 = _Conv2DLayer(filters=gc, kernel_size=1, activation=lrelu_f())
        self.branch2_2 = _Conv2DLayer(filters=gc, kernel_size=(1, 3), activation=lrelu_f())
        self.branch2_3 = _Conv2DLayer(filters=gc, kernel_size=3, activation=None, dilation_rate=3)

        self.branch3_1 = _Conv2DLayer(filters=gc, kernel_size=1, activation=lrelu_f())
        self.branch3_2 = _Conv2DLayer(filters=gc, kernel_size=(3, 1), activation=lrelu_f())
        self.branch3_3 = _Conv2DLayer(filters=gc, kernel_size=3, activation=None, dilation_rate=3)

        self.branch4_1 = _Conv2DLayer(filters=gc //2, kernel_size=1, activation=lrelu_f())
        self.branch4_2 = _Conv2DLayer(filters=(gc //4) * 3, kernel_size=(1, 3), activation=lrelu_f())
        self.branch4_3 = _Conv2DLayer(filters=gc, kernel_size=(3, 1), activation=lrelu_f())
        self.branch4_4 = _Conv2DLayer(filters=gc, kernel_size=3, activation=None, dilation_rate=5)

        self.conv_linear = _Conv2DLayer(filters=nf, kernel_size=1, activation=lrelu_f())
        self.lrelu = lrelu_f()

    def call(self, x):
        shortcut = self.shortcut(x)

        branch1 = self.branch1_1(x)
        branch1 = self.branch1_2(branch1)

        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)
        branch2 = self.branch2_3(branch2)

        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)
        branch3 = self.branch3_3(branch3)

        branch4 = self.branch4_1(x)
        branch4 = self.branch4_2(branch4)
        branch4 = self.branch4_3(branch4)
        branch4 = self.branch4_4(branch4)

        out = tf.concat([branch1, branch2, branch3, branch4], 3)
        out = self.conv_linear(out)

        out = self.lrelu(tf.keras.layers.Add()((out * 0.1, shortcut)))
        return out

class ReceptiveFieldDenseBlock_5C(tf.keras.layers.Layer):
    """Receptive Field Dense Block"""
    def __init__(self, nf=64, gc=16, res_beta=0.2, wd=0., name='RDB5C',
                 **kwargs):
        super(ReceptiveFieldDenseBlock_5C, self).__init__(name=name, **kwargs)
        # gc: growth channel, i.e. intermediate channels
        self.res_beta = res_beta
        self.lrelu = functools.partial(LeakyReLU, alpha=0.2)

        self.rfb1 = ReceptiveFieldBlock(nf=nf, gc=gc, wd=wd)
        self.rfb2 = ReceptiveFieldBlock(nf=nf, gc=gc, wd=wd)
        self.rfb3 = ReceptiveFieldBlock(nf=nf, gc=gc, wd=wd)
        self.rfb4 = ReceptiveFieldBlock(nf=nf, gc=gc, wd=wd)
        self.rfb5 = ReceptiveFieldBlock(nf=nf, gc=gc, wd=wd)

    def call(self, x):
        x1 = self.lrelu()(self.rfb1(x))
        x2 = self.lrelu()(self.rfb2(tf.concat([x, x1], 3)))
        x3 = self.lrelu()(self.rfb3(tf.concat([x, x1, x2], 3)))
        x4 = self.lrelu()(self.rfb4(tf.concat([x, x1, x2, x3], 3)))
        x5 = self.lrelu()(self.rfb5(tf.concat([x, x1, x2, x3, x4], 3)))
        return x5 * self.res_beta + x

class  ResidualOfReceptiveFieldDenseBlock(tf.keras.layers.Layer):
    """Residual in Residual Dense Block"""
    def __init__(self, apply_noise=False, nf=64, gc=16, res_beta=0.2, wd=0., name='RRDB',
                 **kwargs):
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__(name=name, **kwargs)
        self.apply_noise = apply_noise
        self.res_beta = res_beta
        
        self.RFDB1 = ReceptiveFieldDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.RFDB2 = ReceptiveFieldDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.RFDB3 = ReceptiveFieldDenseBlock_5C(nf, gc, res_beta=res_beta, wd=wd)
        self.applynoise = ApplyNoise()

    def call(self, x):
        out = self.RFDB1(x)
        out = self.RFDB2(out)
        out = self.RFDB3(out)
        out = out * self.res_beta + x
        if self.apply_noise:
            out = self.applynoise(out)
        return out

class SubpixelConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, nf=64, gc=16, wd=0.) -> None:
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
        """
        super(SubpixelConvolutionLayer, self).__init__()
        self.lrelu = functools.partial(LeakyReLU, alpha=0.2)
        self.upsample = tf.keras.layers.UpSampling2D((2, 2), interpolation="nearest")
        self.rfb1 = ReceptiveFieldBlock(nf=nf, gc=gc, wd=wd)
        self.conv = Conv2D(nf * 4, kernel_size=3, padding='same')
        self.pixel_shuffle = tf.nn.depth_to_space
        self.rfb2 = ReceptiveFieldBlock(nf=nf, gc=gc, wd=wd)

    def call(self, x):
        out = self.upsample(x)
        out = self.rfb1(out)
        out = self.lrelu()(out)
        out = self.conv(out)
        out = self.pixel_shuffle(out, 2)
        out = self.rfb2(out)
        out = self.lrelu()(out)

        return out

class ApplyNoise(tf.keras.layers.Layer):
  def __init__(self):
    super(ApplyNoise, self).__init__()

  def build(self, input_shape):
    self.channels = input_shape[-1]
    self.channel_wise = self.add_weight("kernel", shape=(self.channels,), trainable=True)

  def call(self, x, training=True):
    if training:
        noise = tf.random.normal(tf.shape(x), dtype=x.dtype)
        return x + noise * self.channel_wise
    else:
        return x

def RRDB_Model_16x(size, channels, cfg_net, gc=32, wd=0., name='RRDB_model'):
    """Residual-in-Residual Dense Block based 16x Model (CIPLAB sequential 4x RRDB network)"""
    nf, nb, apply_noise = cfg_net['nf'], cfg_net['nb'], cfg_net['apply_noise']
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    rrdb_f = functools.partial(ResInResDenseBlock, apply_noise=apply_noise, nf=nf, gc=gc, wd=wd)
    conv_f = functools.partial(Conv2D, kernel_size=3, padding='same',
                               bias_initializer='zeros',
                               kernel_initializer=_kernel_init(),
                               kernel_regularizer=_regularizer(wd))
    rrdb_truck_f_1 = tf.keras.Sequential(
        [rrdb_f(name="RRDB_1_{}".format(i)) for i in range(nb)],
        name='RRDB_trunk_1')
    rrdb_truck_f_2 = tf.keras.Sequential(
        [rrdb_f(name="RRDB_2_{}".format(i)) for i in range(nb)],
        name='RRDB_trunk_2')
    # extraction
    x = inputs = Input([size, size, channels], name='input_image')
    fea = conv_f(filters=nf, name='conv_first_1')(x)
    fea_rrdb = rrdb_truck_f_1(fea)
    trunck = conv_f(filters=nf, name='conv_trunk_1')(fea_rrdb)
    fea = fea + trunck

    # upsampling
    size_fea_h = tf.shape(fea)[1] if size is None else size
    size_fea_w = tf.shape(fea)[2] if size is None else size
    fea_resize = tf.image.resize(fea, [size_fea_h * 2, size_fea_w * 2],
                                 method='nearest', name='upsample_nn_1')
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_1')(fea_resize)
    fea_resize = tf.image.resize(fea, [size_fea_h * 4, size_fea_w * 4],
                                 method='nearest', name='upsample_nn_2')
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_2')(fea_resize)
    fea = conv_f(filters=nf, name='upconv_3')(fea)

    # extraction 2
    fea = conv_f(filters=nf, name='conv_first_2')(fea)
    fea_rrdb = rrdb_truck_f_2(fea)
    trunck = conv_f(filters=nf, name='conv_trunk_2')(fea_rrdb)
    fea = fea + trunck
    fea_resize = tf.image.resize(fea, [size_fea_h * 16, size_fea_w * 16],
                                 method='nearest', name='upsample_nn_final')
    fea = conv_f(filters=nf, activation=lrelu_f(), name='conv_hr')(fea_resize)
    out = conv_f(filters=channels, name='conv_last')(fea)
    return Model(inputs, out, name=name)

def RFB_Model_16x(size, channels, cfg_net, gc=16, wd=0., name='RRDB_model'):
    """Receptive Field Block based 16x Model (OPPO RFB-ESRGAN)"""
    nf, nb_rrdb, nb_rfb, apply_noise = cfg_net['nf'], cfg_net['nb_rrbd'], cfg_net['nb_rfb'], cfg_net['apply_noise']
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    rrdb_f = functools.partial(ResInResDenseBlock, apply_noise=apply_noise, nf=nf, gc=gc, wd=wd)
    rrfb_f = functools.partial(ResidualOfReceptiveFieldDenseBlock, apply_noise=apply_noise, nf=nf, gc=gc, wd=wd)
    conv_f = functools.partial(Conv2D, kernel_size=3, padding='same',
                               bias_initializer='zeros',
                               kernel_initializer=_kernel_init(),
                               kernel_regularizer=_regularizer(wd))
    rrdb_truck = tf.keras.Sequential(
        [rrdb_f(name="RRDB_{}".format(i)) for i in range(nb_rrdb)],
        name='RRDB_trunk')
    rfb_truck = tf.keras.Sequential(
        [rrfb_f(name="RFB_{}".format(i)) for i in range(nb_rfb)],
        name='RFB_trunk')
    # extraction
    x = inputs = Input([size, size, channels], name='input_image')
    fea = conv_f(filters=nf, name='conv_first_1')(x)    # first conv
    fea_rrdb = rrdb_truck(fea)

    fea_rfb = rfb_truck(fea_rrdb)
    fea = fea + fea_rfb
    trunck = ReceptiveFieldBlock(nf=nf, gc=gc, wd=wd, name='conv_trunk_1')(fea_rfb)
    
    # upsampling
    up1 = SubpixelConvolutionLayer(nf, gc, wd)(trunck)  # 4x down
    up2 = SubpixelConvolutionLayer(nf, gc, wd)(up1)     # hr res
    out = conv_f(filters=nf, name='conv_last_1')(up2)
    out = lrelu_f()(out)
    out = conv_f(filters=channels, name='conv_first_2')(out)
    return Model(inputs, out, name=name)

def RRDB_Model(size, channels, cfg_net, gc=32, wd=0., name='RRDB_model'):
    """Residual-in-Residual Dense Block based Model """
    nf, nb, apply_noise = cfg_net['nf'], cfg_net['nb'], cfg_net['apply_noise']
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    rrdb_f = functools.partial(ResInResDenseBlock, apply_noise=apply_noise, nf=nf, gc=gc, wd=wd)
    conv_f = functools.partial(Conv2D, kernel_size=3, padding='same',
                               bias_initializer='zeros',
                               kernel_initializer=_kernel_init(),
                               kernel_regularizer=_regularizer(wd))
    rrdb_truck_f = tf.keras.Sequential(
        [rrdb_f(name="RRDB_{}".format(i)) for i in range(nb)],
        name='RRDB_trunk')

    # extraction
    x = inputs = Input([size, size, channels], name='input_image')
    fea = conv_f(filters=nf, name='conv_first')(x)
    fea_rrdb = rrdb_truck_f(fea)
    trunck = conv_f(filters=nf, name='conv_trunk')(fea_rrdb)
    fea = fea + trunck

    # upsampling
    size_fea_h = tf.shape(fea)[1] if size is None else size
    size_fea_w = tf.shape(fea)[2] if size is None else size
    fea_resize = tf.image.resize(fea, [size_fea_h * 2, size_fea_w * 2],
                                 method='nearest', name='upsample_nn_1')
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_1')(fea_resize)
    fea_resize = tf.image.resize(fea, [size_fea_h * 4, size_fea_w * 4],
                                 method='nearest', name='upsample_nn_2')
    fea = conv_f(filters=nf, activation=lrelu_f(), name='upconv_2')(fea_resize)
    fea = conv_f(filters=nf, activation=lrelu_f(), name='conv_hr')(fea)
    out = conv_f(filters=channels, name='conv_last')(fea)

    return Model(inputs, out, name=name)


def DiscriminatorVGG128(size, channels, nf=64, wd=0., scale=4,
                        name='Discriminator_VGG_128', refgan=False):
    """Discriminator VGG 128"""
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    conv_k3s1_f = functools.partial(Conv2D,
                                    kernel_size=3, strides=1, padding='same',
                                    kernel_initializer=_kernel_init(),
                                    kernel_regularizer=_regularizer(wd))
    conv_k4s2_f = functools.partial(Conv2D,
                                    kernel_size=4, strides=2, padding='same',
                                    kernel_initializer=_kernel_init(),
                                    kernel_regularizer=_regularizer(wd))
    dese_f = functools.partial(Dense, kernel_regularizer=_regularizer(wd))

    x = inputs = Input(shape=(size, size, channels))

    if refgan:
        ref = Input(shape=(size//scale,size//scale,channels))
        ref_up = tf.keras.layers.experimental.preprocessing.Resizing(size, size, interpolation='bicubic')(ref)
        x = tf.keras.layers.concatenate([x, ref_up])

    x = conv_k3s1_f(filters=nf, name='conv0_0')(x)
    x = conv_k4s2_f(filters=nf, use_bias=False, name='conv0_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn0_1')(x))

    x = conv_k3s1_f(filters=nf * 2, use_bias=False, name='conv1_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn1_0')(x))
    x = conv_k4s2_f(filters=nf * 2, use_bias=False, name='conv1_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn1_1')(x))

    x = conv_k3s1_f(filters=nf * 4, use_bias=False, name='conv2_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn2_0')(x))
    x = conv_k4s2_f(filters=nf * 4, use_bias=False, name='conv2_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn2_1')(x))

    x = conv_k3s1_f(filters=nf * 8, use_bias=False, name='conv3_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn3_0')(x))
    x = conv_k4s2_f(filters=nf * 8, use_bias=False, name='conv3_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn3_1')(x))

    x = conv_k3s1_f(filters=nf * 8, use_bias=False, name='conv4_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn4_0')(x))
    x = conv_k4s2_f(filters=nf * 8, use_bias=False, name='conv4_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn4_1')(x))

    x = Flatten()(x)
    x = dese_f(units=100, activation=lrelu_f(), name='linear1')(x)
    out = dese_f(units=1, name='linear2')(x)
    if refgan:
        return Model([inputs,ref], out, name=name)
    else:
        return Model(inputs, out, name=name)

def DiscriminatorVGG512(size, channels, nf=64, wd=0., scale=4,
                        name='Discriminator_VGG_128', refgan=False):
    """Discriminator VGG 128"""
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    conv_k3s1_f = functools.partial(Conv2D,
                                    kernel_size=3, strides=1, padding='same',
                                    kernel_initializer=_kernel_init(),
                                    kernel_regularizer=_regularizer(wd))
    conv_k3s2_f = functools.partial(Conv2D,
                                    kernel_size=3, strides=2, padding='same',
                                    kernel_initializer=_kernel_init(),
                                    kernel_regularizer=_regularizer(wd))
    dese_f = functools.partial(Dense, kernel_regularizer=_regularizer(wd))

    x = inputs = Input(shape=(size, size, channels))

    if refgan:
        ref = Input(shape=(size//scale,size//scale,channels))
        ref_up = tf.keras.layers.experimental.preprocessing.Resizing(size, size, interpolation='bicubic')(ref)
        x = tf.keras.layers.concatenate([x, ref_up])

    x = conv_k3s1_f(filters=nf, name='conv0_0')(x)
    x = lrelu_f()(x)

    x = conv_k3s2_f(filters=nf, use_bias=False, name='conv0_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn0_1')(x))

    x = conv_k3s1_f(filters=nf * 2, use_bias=False, name='conv1_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn1_0')(x))
    x = conv_k3s2_f(filters=nf * 2, use_bias=False, name='conv1_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn1_1')(x))

    x = conv_k3s1_f(filters=nf * 4, use_bias=False, name='conv2_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn2_0')(x))
    x = conv_k3s2_f(filters=nf * 4, use_bias=False, name='conv2_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn2_1')(x))

    x = conv_k3s1_f(filters=nf * 8, use_bias=False, name='conv3_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn3_0')(x))
    x = conv_k3s2_f(filters=nf * 8, use_bias=False, name='conv3_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn3_1')(x))

    x = conv_k3s1_f(filters=nf * 8, use_bias=False, name='conv4_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn4_0')(x))
    x = conv_k3s2_f(filters=nf * 8, use_bias=False, name='conv4_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn4_1')(x))

    x = Flatten()(x)
    x = dese_f(units=100, activation=lrelu_f(), name='linear1')(x)
    out = dese_f(units=1, name='linear2')(x)
    if refgan:
        return Model([inputs,ref], out, name=name)
    else:
        return Model(inputs, out, name=name)
