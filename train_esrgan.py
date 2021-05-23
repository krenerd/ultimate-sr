from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from modules.models import RRDB_Model, RRDB_Model_16x, RFB_Model_16x, DiscriminatorVGG128, DiscriminatorVGG512
from modules.lr_scheduler import MultiStepLR
from modules.losses import (PixelLoss, ContentLoss, DiscriminatorLoss, gradient_penalty,
                            GeneratorLoss, PixelLossDown)
from modules.utils import (load_yaml, load_dataset, ProgressBar,
                           set_memory_growth, load_val_dataset)
from evaluate import evaluate_dataset

flags.DEFINE_string('cfg_path', './configs/esrgan.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')


def main(_):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    if cfg['network_G']['name']=='RRDB':    # ESRGAN 4x
        generator = RRDB_Model(None, cfg['ch_size'], cfg['network_G'])
    elif cfg['network_G']['name']=='RRDB_CIPLAB':
        generator = RRDB_Model_16x(None, cfg['ch_size'], cfg['network_G'])
    elif cfg['network_G']['name']=='RFB_ESRGAN':
        generator = RFB_Model_16x(None, cfg['ch_size'], cfg['network_G'])
    generator.summary(line_length=80)

    if cfg['network_G']['name']=='RRDB':
        discriminator = DiscriminatorVGG128(cfg['gt_size'], cfg['ch_size'], scale=cfg['scale'], refgan=cfg['refgan'])
    elif cfg['network_G']['name']=='RRDB_CIPLAB':
        pass
    elif cfg['network_G']['name']=='RFB_ESRGAN':
        discriminator = DiscriminatorVGG512(cfg['gt_size'], cfg['ch_size'], scale=cfg['scale'], refgan=cfg['refgan'])

    discriminator.summary(line_length=80)

    # load dataset
    train_dataset = load_dataset(cfg, 'train_dataset', shuffle=False)
    set5_dataset = load_val_dataset(cfg, 'set5')
    set14_dataset = load_val_dataset(cfg, 'set14')
    if 'DIV8K' in cfg['test_dataset']:
        print('[*] Loading test dataset.')
        DIV8K_val = load_val_dataset(cfg, 'DIV8K', crop_centor=cfg['test_dataset']['DIV8K_crop_centor'])

    # define optimizer
    learning_rate_G = MultiStepLR(cfg['lr_G'], cfg['lr_steps'], cfg['lr_rate'])
    learning_rate_D = MultiStepLR(cfg['lr_D'], cfg['lr_steps'], cfg['lr_rate'])
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=learning_rate_G,
                                           beta_1=cfg['adam_beta1_G'],
                                           beta_2=cfg['adam_beta2_G'])
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=learning_rate_D,
                                           beta_1=cfg['adam_beta1_D'],
                                           beta_2=cfg['adam_beta2_D'])

    # define losses function
    if cfg['cycle_mse']:
        pixel_loss_fn = PixelLossDown(criterion=cfg['pixel_criterion'], 
                                    lr_size=(cfg['gt_size']//cfg['scale'], cfg['gt_size']//cfg['scale']))
    else:
        pixel_loss_fn = PixelLoss(criterion=cfg['pixel_criterion'])
    fea_loss_fn = ContentLoss(criterion=cfg['feature_criterion'])
    gen_loss_fn = GeneratorLoss(gan_type=cfg['gan_type'])
    dis_loss_fn = DiscriminatorLoss(gan_type=cfg['gan_type'])

    # load checkpoint
    checkpoint_dir = cfg['log_dir'] + '/checkpoints/'
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     optimizer_G=optimizer_G,
                                     optimizer_D=optimizer_D,
                                     model=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('[*] load ckpt from {} at step {}.'.format(
            manager.latest_checkpoint, checkpoint.step.numpy()))
    else: # if checkpoint file doesn't exist
        if cfg['pretrain_dir'] is not None:    # load from pretrained model
            pretrain_dir = cfg['pretrain_dir'] + '/checkpoints/'
            if tf.train.latest_checkpoint(pretrain_dir):
                checkpoint.restore(tf.train.latest_checkpoint(pretrain_dir))
                checkpoint.step.assign(0)
                print("[*] training from pretrain model {}.".format(
                    pretrain_dir))
            else:
                print("[*] cannot find pretrain model {}.".format(
                    pretrain_dir))
        else:
            print("[*] training from scratch.")

    # define training step function
    @tf.function
    def train_step(lr, hr):
        with tf.GradientTape(persistent=True) as tape:
            sr = generator(lr, training=True)
            if cfg['refgan']:
                hr_output = discriminator([hr, lr], training=True)
                sr_output = discriminator([sr, lr], training=True)
            else: 
                hr_output = discriminator(hr, training=True)
                sr_output = discriminator(sr, training=True)

            losses_G = {}
            losses_D = {}
            losses_G['reg'] = tf.reduce_sum(generator.losses)
            losses_D['reg'] = tf.reduce_sum(discriminator.losses)
            
            if cfg['w_gan'] > 0.0 and cfg['gan_type']=='wgan-gp':  # add GP loss
                losses_D['gp'] = gradient_penalty(discriminator, hr, sr, lr, refgan=cfg['refgan']) * cfg['gp_weight']

            if cfg['w_pixel'] > 0.0:
                losses_G['pixel'] = cfg['w_pixel'] * pixel_loss_fn(hr, sr)
            if cfg['w_feature'] > 0.0:
                losses_G['feature'] = cfg['w_feature'] * fea_loss_fn(hr, sr)
            if cfg['w_gan'] > 0.0:
                losses_G['gan'] = cfg['w_gan'] * gen_loss_fn(hr_output, sr_output)
                losses_D['gan'] = dis_loss_fn(hr_output, sr_output)
                
            total_loss_G = tf.add_n([l for l in losses_G.values()])
            total_loss_D = tf.add_n([l for l in losses_D.values()])

        grads_G = tape.gradient(
            total_loss_G, generator.trainable_variables)
        grads_D = tape.gradient(
            total_loss_D, discriminator.trainable_variables)
        optimizer_G.apply_gradients(
            zip(grads_G, generator.trainable_variables))
        optimizer_D.apply_gradients(
            zip(grads_D, discriminator.trainable_variables))

        return total_loss_G, total_loss_D, losses_G, losses_D

    # training loop
    summary_writer = tf.summary.create_file_writer(cfg['log_dir']+'/logs')
    prog_bar = ProgressBar(cfg['niter'], checkpoint.step.numpy())
    remain_steps = max(cfg['niter'] - checkpoint.step.numpy(), 0)

    for _ in range(remain_steps):
        lr, hr = train_dataset()

        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss_G, total_loss_D, losses_G, losses_D = train_step(lr, hr)

        prog_bar.update(
            "loss_G={:.4f}, loss_D={:.4f}, lr_G={:.1e}, lr_D={:.1e}".format(
                total_loss_G.numpy(), total_loss_D.numpy(),
                optimizer_G.lr(steps).numpy(), optimizer_D.lr(steps).numpy()))

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar(
                    'loss_G/total_loss', total_loss_G, step=steps)
                tf.summary.scalar(
                    'loss_D/total_loss', total_loss_D, step=steps)
                for k, l in losses_G.items():
                    tf.summary.scalar('loss_G/{}'.format(k), l, step=steps)
                for k, l in losses_D.items():
                    tf.summary.scalar('loss_D/{}'.format(k), l, step=steps)

                tf.summary.scalar(
                    'learning_rate_G', optimizer_G.lr(steps), step=steps)
                tf.summary.scalar(
                    'learning_rate_D', optimizer_D.lr(steps), step=steps)

        if steps % cfg['save_steps'] == 0:
            manager.save()
            print("\n[*] save ckpt file at {}".format(
                manager.latest_checkpoint))

            # log results on test data
            set5_logs = evaluate_dataset(set5_dataset, generator, cfg)
            set14_logs = evaluate_dataset(set14_dataset, generator, cfg)
            if 'DIV8K' in cfg['test_dataset']:
                DIV8K_logs = evaluate_dataset(DIV8K_val, generator, cfg)

            with summary_writer.as_default():
                if cfg['logging']['psnr']:
                    tf.summary.scalar('set5/psnr', set5_logs['psnr'], step=steps)
                    tf.summary.scalar('set14/psnr', set14_logs['psnr'], step=steps)
                    if 'DIV8K' in cfg['test_dataset']:
                        tf.summary.scalar('DIV8K/psnr', DIV8K_logs['psnr'], step=steps)

                if cfg['logging']['ssim']:
                    tf.summary.scalar('set5/ssim', set5_logs['ssim'], step=steps)
                    tf.summary.scalar('set14/ssim', set14_logs['ssim'], step=steps)
                    if 'DIV8K' in cfg['test_dataset']:
                        tf.summary.scalar('DIV8K/psnr', DIV8K_logs['psnr'], step=steps)

                if cfg['logging']['lpips']:
                    tf.summary.scalar('set5/lpips', set5_logs['lpips'], step=steps)
                    tf.summary.scalar('set14/lpips', set14_logs['lpips'], step=steps)
                    if 'DIV8K' in cfg['test_dataset']:
                        tf.summary.scalar('DIV8K/lpips', DIV8K_logs['lpips'], step=steps)

                if cfg['logging']['plot_samples']:
                    tf.summary.image("set5/samples", [set5_logs['samples']], step=steps)
                    tf.summary.image("set14/samples", [set14_logs['samples']], step=steps)
                    if 'DIV8K' in cfg['test_dataset']:
                        tf.summary.image("DIV8K/samples", [DIV8K_logs['samples']], step=steps)
                        
    print("\n [*] training done!")


if __name__ == '__main__':
    app.run(main)
