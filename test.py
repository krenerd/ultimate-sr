from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf

from modules.resizing import imresize_np
from modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)


flags.DEFINE_string('model', '', 'model weight path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_integer('scale', 4, 'upscaling scale image')
flags.DEFINE_bool('down_up', False, 'downscale->upscale for testing?')

def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    # define network
    print("[*] Loading weights from {FLAGS.model}")
    model=tf.keras.models.load_model(FLAGS.model)
    # evaluation
    if FLAGS.down_up:
        print("[*] Processing on single image {}".format(FLAGS.img_path))
        raw_img = cv2.imread(FLAGS.img_path)
        lr_img, hr_img = create_lr_hr_pair(raw_img, FLAGS.scale)

        sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
        bic_img = imresize_np(lr_img, FLAGS.scale).astype(np.uint8)

        str_format = "[{}] PSNR/SSIM: SR={:.2f}db/{:.2f}"
        print(str_format.format(
            os.path.basename(FLAGS.img_path),
            calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
            calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))))
        result_img_path = './Bic_SR_HR_' + os.path.basename(FLAGS.img_path)
        print("[*] write the result image {}".format(result_img_path))
        results_img = np.concatenate((bic_img, sr_img, hr_img), 1)
        cv2.imwrite(result_img_path, results_img)
    else:
        print("[*] Processing on single image {}".format(FLAGS.img_path))
        lr_img = cv2.imread(FLAGS.img_path)
        sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))

        result_img_path = './SR_' + os.path.basename(FLAGS.img_path)
        print("[*] write the result image {}".format(result_img_path))
        cv2.imwrite(result_img_path, sr_img)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
