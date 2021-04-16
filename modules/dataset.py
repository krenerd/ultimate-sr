import tensorflow as tf


def _parse_tfrecord(gt_size, scale, using_bin, using_flip, using_rot):
    def parse_tfrecord(tfrecord):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image)

        lr_img, hr_img = _transform_images(
            gt_size, scale, using_flip, using_rot)(image)

        return lr_img, hr_img
    return parse_tfrecord


def _transform_images(gt_size, scale, using_flip, using_rot):
    def transform_images(img):
        # randomly crop
        hr_img = tf.image.random_crop(img, (gt_size,gt_size,3))
        lr_img = tf.image.resize(hr_img,(gt_size//scale, gt_size//scale), method = tf.image.ResizeMethod.BICUBIC)
        # randomly left-right flip
        if using_flip:
            flip_case = tf.random.uniform([1], 0, 2, dtype=tf.int32)
            
            def flip_func(): return (tf.image.flip_left_right(lr_img),
                                     tf.image.flip_left_right(hr_img))
            lr_img, hr_img = tf.case(
                [(tf.equal(flip_case, 0), flip_func)],
                default=lambda: (lr_img, hr_img))

        # randomly rotation
        if using_rot:
            rot_case = tf.random.uniform([1], 0, 4, dtype=tf.int32)
            def rot90_func(): return (tf.image.rot90(lr_img, k=1),
                                      tf.image.rot90(hr_img, k=1))
            def rot180_func(): return (tf.image.rot90(lr_img, k=2),
                                       tf.image.rot90(hr_img, k=2))
            def rot270_func(): return (tf.image.rot90(lr_img, k=3),
                                       tf.image.rot90(hr_img, k=3))
            lr_img, hr_img = tf.case(
                [(tf.equal(rot_case, 0), rot90_func),
                 (tf.equal(rot_case, 1), rot180_func),
                 (tf.equal(rot_case, 2), rot270_func)],
                default=lambda: (lr_img, hr_img))

        # scale to [0, 1]
        lr_img = lr_img / 255
        hr_img = hr_img / 255

        return lr_img, hr_img
    return transform_images


def load_tfrecord_dataset(tfrecord_name, batch_size, gt_size,
                          scale, using_bin=False, using_flip=False,
                          using_rot=False, shuffle=True, buffer_size=10240):
    """load dataset from tfrecord"""
    train_path = tf.data.Dataset.list_files(tfrecord_name+'/*.png')

    if shuffle:
        train_path = train_path.shuffle(buffer_size=buffer_size)
    dataset = train_path.map(
        _parse_tfrecord(gt_size, scale, using_bin, using_flip, using_rot),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
