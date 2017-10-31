import argparse
import logging
import os.path as osp
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import SHAPENET_IM
from evaluate import eval_l1_err, print_depth_stats
from loader import pad_batch
from models import grid_nets, im_nets, model_dlsm
from mvnet import MVNet
from ops import conv_rnns
from shapenet import ShapeNet
from tensorboard_logging import TensorboardLogger
from utils import get_session_config, init_logging, mkdir_p, process_args


def tensorboard_log(stats, tbd, step):
    tbd.log_scalar('masked_l1_err', np.mean(stats), step)


def validate(args, checkpoint):
    net = MVNet(
        vmin=-0.5,
        vmax=0.5,
        vox_bs=args.val_batch_size,
        im_bs=args.val_im_batch,
        grid_size=args.nvox,
        im_h=args.im_h,
        im_w=args.im_w,
        mode="TEST",
        norm=args.norm)

    im_dir = SHAPENET_IM

    # Setup network
    net = model_dlsm(
        net,
        im_nets[args.im_net],
        grid_nets[args.grid_net],
        conv_rnns[args.rnn],
        im_skip=args.im_skip,
        ray_samples=args.ray_samples,
        sepup=args.sepup,
        proj_x=args.proj_x,
        proj_last=True)
    sess = tf.Session(config=get_session_config())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    coord = tf.train.Coordinator()

    # Init dataset
    dset = ShapeNet(im_dir=im_dir, split_file=args.val_split_file, rng_seed=1)
    mids = dset.get_smids('val')
    logging.info('Validating %d models', len(mids))
    items = ['shape_id', 'model_id', 'im', 'K', 'R', 'depth']
    dset.init_queue(
        mids,
        args.val_im_batch,
        items,
        coord,
        nepochs=1,
        qsize=32,
        nthreads=args.prefetch_threads)

    # Init stats
    l1_err = []

    # Testing loop
    pbar = tqdm(desc='Validating', total=len(mids))
    deq_mids, deq_sids = [], []
    try:
        while not coord.should_stop():
            batch_data = dset.next_batch(items, net.batch_size)
            if batch_data is None:
                continue
            deq_sids.append(batch_data['shape_id'])
            deq_mids.append(batch_data['model_id'])
            num_batch_items = batch_data['K'].shape[0]
            batch_data = pad_batch(batch_data, args.val_batch_size)
            feed_dict = {
                net.K: batch_data['K'],
                net.Rcam: batch_data['R'],
                net.ims: batch_data['im']
            }
            pred = sess.run(net.depth_out, feed_dict=feed_dict)
            batch_err = eval_l1_err(pred[:num_batch_items],
                                    batch_data['depth'][:num_batch_items])

            l1_err.extend(batch_err)
            pbar.update(num_batch_items)
    except Exception, e:
        logger.error(repr(e))
        dset.close_queue(e)
    finally:
        pbar.close()
        sess.close()
        logger.info('Validation completed')
        coord.join()

    deq_mids = np.concatenate(deq_mids, axis=0)[..., 0].tolist()
    deq_sids = np.concatenate(deq_sids, axis=0)[..., 0].tolist()

    # Print statistics and save to file
    stats, stats_table = print_depth_stats(zip(deq_sids, deq_mids), l1_err)

    return stats, stats_table


def parse_args():
    parser = argparse.ArgumentParser(description='Options for MVNet')
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--val_im_batch', type=int, default=4)
    parser.add_argument('--loglevel', type=str, default='info')
    parser.add_argument(
        '--val_split_file', type=str, default='data/splits.json')
    parser.add_argument('--prefetch_threads', type=int, default=2)
    parser.add_argument('--sleep_time', type=int, default=15)
    args = process_args(parser)
    return args


if __name__ == '__main__':
    args = parse_args()
    init_logging(args.loglevel)
    logger = logging.getLogger('mview3d.' + __name__)
    logger.info('Starting validation @ %s', args.log)
    # Initialize tensorboard logger
    mkdir_p(osp.join(args.log, 'val'))
    tbd_logger = TensorboardLogger(log_dir=osp.join(args.log, 'val'))
    processed = []
    while True:
        tf.reset_default_graph()
        latest_checkpoint = tf.train.latest_checkpoint(args.log)
        if latest_checkpoint is None or latest_checkpoint in processed:
            time.sleep(args.sleep_time * 60)
            logger.info('Checking for new checkpoints')
            continue
        step = int(osp.basename(latest_checkpoint).split('-')[-1])
        logger.info('Validate %s', latest_checkpoint)
        val_stats, table = validate(args, latest_checkpoint)
        tensorboard_log(val_stats, tbd_logger, step)
        processed.append(latest_checkpoint)
        logging.info(table)
        if step >= args.niters:
            logging.info('Finished training/validation for %d iterations',
                         args.niters)
            break
