import argparse
import logging
import os.path as osp
import time
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm

from config import SHAPENET_IM, SHAPENET_VOX
from models import grid_nets, im_nets, model_vlsm
from mvnet import MVNet
from ops import conv_rnns, image_sum, loss_ce, repeat_tensor, voxel_sum
from shapenet import ShapeNet
from utils import (Timer, get_session_config, init_logging, mkdir_p,
                   process_args, write_args)


def train(net):

    net.gt_vox = tf.placeholder(tf.float32, net.vox_tensor_shape)
    # Add loss to the graph
    net.loss = loss_ce(net.pred_vox, repeat_tensor(net.gt_vox, net.im_batch))

    _t_dbg = Timer()

    # Add optimizer
    global_step = tf.Variable(0, trainable=False, name='global_step')
    decay_lr = tf.train.exponential_decay(
        args.lr,
        global_step,
        args.decay_steps,
        args.decay_rate,
        staircase=True)
    lr_sum = tf.summary.scalar('lr', decay_lr)
    optim = tf.train.AdamOptimizer(decay_lr).minimize(net.loss, global_step)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Add summaries for training
    net.loss_sum = tf.summary.scalar('loss', net.loss)
    net.vox_sum = voxel_sum(net)
    net.im_sum = image_sum(net.ims, net.batch_size, net.im_batch)
    merged = tf.summary.merge([net.vox_sum, net.im_sum])
    merged_scalars = tf.summary.merge([net.loss_sum, lr_sum])

    # Initialize dataset
    coord = tf.train.Coordinator()
    dset = ShapeNet(
        im_dir=im_dir,
        split_file=args.split_file,
        rng_seed=args.rng_seed,
        vox_dir=vox_dir)
    mids = dset.get_smids('train')
    logger.info('Training with %d models', len(mids))
    
    items = ['im', 'K', 'R', 'vol']

    dset.init_queue(
        mids,
        net.im_batch,
        items,
        coord,
        qsize=32,
        nthreads=args.prefetch_threads)

    iters = 0
    # Training loop
    pbar = tqdm(desc='Training Vox-LSM', total=args.niters)
    with tf.Session(config=get_session_config()) as sess:
        sum_writer = tf.summary.FileWriter(log_dir, sess.graph)
        if args.ckpt is not None:
            logger.info('Restoring from %s', args.ckpt)
            saver.restore(sess, args.ckpt)
        else:
            sess.run(init_op)

        try:
            while True:
                iters += 1
                _t_dbg.tic()
                batch_data = dset.next_batch(items, net.batch_size)
                logging.debug('Data read time - %.3fs', _t_dbg.toc())
                feed_dict = {
                    net.K: batch_data['K'],
                    net.Rcam: batch_data['R'],
                    net.gt_vox: batch_data['vol']
                }
                feed_dict[net.ims] = batch_data['im']

                _t_dbg.tic()
                if args.run_trace and (iters % args.sum_iters == 0 or
                                       iters == 1 or iters == args.niters):
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    step_, _, merged_scalars_ = sess.run(
                        [global_step, optim, merged_scalars],
                        feed_dict=feed_dict,
                        options=run_options,
                        run_metadata=run_metadata)
                    sum_writer.add_run_metadata(run_metadata, 'step%d' % step_)
                else:
                    step_, _, merged_scalars_ = sess.run(
                        [global_step, optim, merged_scalars],
                        feed_dict=feed_dict)

                logging.debug('Net time - %.3fs', _t_dbg.toc())

                sum_writer.add_summary(merged_scalars_, step_)
                if iters % args.sum_iters == 0 or iters == 1 or iters == args.niters:
                    image_sum_, step_ = sess.run(
                        [merged, global_step], feed_dict=feed_dict)
                    sum_writer.add_summary(image_sum_, step_)

                if iters % args.ckpt_iters == 0 or iters == args.niters:
                    save_f = saver.save(
                        sess,
                        osp.join(log_dir, 'mvnet'),
                        global_step=global_step)
                    logger.info(' Model checkpoint - {:s} '.format(save_f))

                pbar.update(1)
                if iters >= args.niters:
                    break
        except Exception, e:
            logging.error(repr(e))
            dset.close_queue(e)
        finally:
            pbar.close()
            logger.info('Training completed')
            dset.close_queue()
            coord.join()


def parse_args():
    parser = argparse.ArgumentParser(description='Options for MVNet')
    parser.add_argument('--argsjs', type=str, default=None)
    parser.add_argument('--logdir', type=str, default='./log')
    parser.add_argument('--loglevel', type=str, default='info')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--im_batch', type=int, default=4)
    parser.add_argument('--prefetch_threads', type=int, default=2)
    parser.add_argument('--im_h', type=int, default=224)
    parser.add_argument('--im_w', type=int, default=224)
    parser.add_argument('--nvox', type=int, default=32)
    parser.add_argument('--norm', type=str, default='IN')
    parser.add_argument(
        '--im_net', type=str, default='unet', choices=im_nets.keys())
    parser.add_argument(
        '--grid_net', type=str, default='unet32', choices=grid_nets.keys())
    parser.add_argument(
        '--rnn', type=str, default='gru', choices=conv_rnns.keys())
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay_rate', type=float, default=1)
    parser.add_argument('--decay_steps', type=int, default=10000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--niters', type=int, default=10000)
    parser.add_argument('--sum_iters', type=int, default=50)
    parser.add_argument('--ckpt_iters', type=int, default=5000)
    parser.add_argument('--rng_seed', type=int, default=0)
    parser.add_argument('--run_trace', action="store_true")
    args = process_args(parser)
    return args


if __name__ == '__main__':
    key = time.strftime("%Y-%m-%d_%H%M%S")
    args = parse_args()
    init_logging(args.loglevel)
    logger = logging.getLogger('mview3d.' + __name__)
    vox_dir = SHAPENET_VOX[args.nvox]
    im_dir = SHAPENET_IM
    if args.ckpt is None:
        log_dir = osp.join(args.logdir, key, 'train')
    else:
        log_dir = args.logdir

    # Initialize network parameters
    mvnet = MVNet(
        vmin=-0.5,
        vmax=0.5,
        vox_bs=args.batch_size,
        im_bs=args.im_batch,
        grid_size=args.nvox,
        im_h=args.im_h,
        im_w=args.im_w,
        mode="TRAIN",
        norm=args.norm)

    # Define graph
    mvnet = model_vlsm(mvnet, im_nets[args.im_net], grid_nets[args.grid_net],
                       conv_rnns[args.rnn])

    # Set things up
    mkdir_p(log_dir)
    write_args(args, osp.join(log_dir, 'args.json'))
    logger.info('Logging to {:s}'.format(log_dir))
    logger.info('\nUsing args:')
    pprint(vars(args))
    mvnet.print_net()

    # Train loop
    train(mvnet)
