import tensorflow as tf
import numpy as np
import cv2
import tqdm
import os
from modules.resizing import imresize_np

def load_valid_dataset( data_path, scale=4, crop_centor=0):
    # evaluate the model in various datasets for various methods
    @tf.function()
    def read_image(path):
        # read image in graph-mode
        image = tf.io.read_file(path)
        image = image = tf.io.decode_image(image, expand_animations = False)
        return image / 255
        
    def generate_val_data(image):
        # Returns (LR, HR)
        image=tf.dtypes.cast(image, tf.float32)

        if crop_centor > 0:  
            image = tf.keras.layers.experimental.preprocessing.CenterCrop(crop_centor, crop_centor)([image])[0]
        else:
            image = tf.image.resize(image,((image.shape[0]//scale) * scale,(image.shape[1]//scale)*scale) ,
                                method=tf.image.ResizeMethod.BICUBIC).numpy()
        lr = imresize_np(image, 1/scale)    # generate lr copy
        return lr,image
    path_list=[os.path.join(data_path, x) for x in sorted(os.listdir(data_path))]   # sort data path and read in order
    path_list = tf.data.Dataset.from_tensor_slices(path_list)

    dataset = path_list.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    im_list=[]
    for image in tqdm.tqdm(dataset, position=0, leave=True):
        im_list.append(generate_val_data(image))
    return im_list

def read_img(path):
    image = tf.io.read_file(path)
    image = image = tf.io.decode_image(image, expand_animations = False)
    return image

def _transform_images(gt_size, scale, using_flip, using_rot, detect_blur):
    def transform_images(img):
        # randomly crop
        hr_img = tf.image.random_crop(img, (gt_size,gt_size,3))
        lr_img = imresize_np(hr_img, 1/scale)
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
        # detect blurry
        if detect_blur:
            var=cv2.Laplacian(hr_img.numpy(), cv2.CV_32F).var()
            if var < 100: #Blurry Image
                return None 
                
        # scale to [0, 1]
        lr_img = lr_img / 255
        hr_img = hr_img / 255

        return lr_img, hr_img
    return transform_images

def generate_patches(im, buffer, patch_per_image, gt_size, scale, using_bin, 
                    using_flip, using_rot, detect_blur):
    for _ in range(patch_per_image):
        res = _transform_images(gt_size, scale, using_flip, using_rot, detect_blur)(im)
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
def load_tfrecord_dataset(tfrecord_name, batch_size, gt_size, scale,
                          using_bin=False, using_flip=False, detect_blur=True,
                          using_rot=False, shuffle=True, buffer_size=1024,
                          patch_per_image=128):
    """returns function for loading data"""
    buffer=[]

    f_read_image=lambda im:generate_patches(im, buffer, patch_per_image,gt_size, scale,
        using_bin, using_flip, using_rot, detect_blur)
    train_path=[os.path.join(tfrecord_name, x) for x in os.listdir(tfrecord_name)]
    train_path = tf.data.Dataset.from_tensor_slices(train_path).shuffle(len(train_path))

    read_image = train_path.map(read_img, num_parallel_calls=tf.data.AUTOTUNE)
    image_iter = iter(read_image.repeat())
    return lambda :load_data_batch(buffer, image_iter, f_read_image, batch_size, buffer_size)
