import tensorflow as tf
import numpy as np
def read_img(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image)
    return image

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

def generate_patches(im, buffer, patch_per_image, gt_size, scale, using_bin, using_flip, using_rot):
    for _ in range(patch_per_image):
        res = _transform_images(gt_size, scale, using_flip, using_rot)(im)
        if not res is None:
            buffer.append(res)

def load_data_batch(buffer, image_iter, _generate_patches, batch_size, buffer_size):
    lr_list, hr_list = [], []
    # Fill the buffer until size >= `buffer_size`
    while len(buffer) < buffer_size:
        im = image_iter.get_next()
        _generate_patches(im)
    # Load batch_size patches from buffer
    for _ in range(batch_size):
        idx = np.random.randint(len(buffer))

        lr_list.append(buffer[idx][0])
        hr_list.append(buffer[idx][1])

        del buffer[idx]
        
    return (np.array(lr_list), np.array(hr_list))
def load_tfrecord_dataset(tfrecord_name, batch_size, gt_size,
                          scale, using_bin=False, using_flip=False,
                          using_rot=False, shuffle=True, buffer_size=1024,
                          patch_per_image=128):
    """returns function for loading data"""
    buffer=[]

    f_read_image=lambda im:generate_patches(im, buffer, patch_per_image,
        gt_size, scale, using_bin, using_flip, using_rot)

    train_path = tf.data.Dataset.list_files(tfrecord_name+'/*.png')
    read_image = train_path.map(read_img, num_parallel_calls=tf.data.AUTOTUNE)
    image_iter = iter(read_image.repeat())
    return lambda :load_data_batch(buffer, image_iter, f_read_image, batch_size, buffer_size)
