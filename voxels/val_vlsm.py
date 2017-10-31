import argparse
import logging
import os.path as osp
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import SHAPENET_IM, SHAPENET_VOX
from evaluate import eval_seq_iou, init_iou, print_iou_stats, update_iou
from loader import pad_batch
from models import grid_nets, im_nets, model_vlsm
from mvnet import MVNet
from ops import conv_rnns
from shapenet import ShapeNet
from tensorboard_logging import TensorboardLogger
from utils import get_session_config, init_logging, mkdir_p, process_args


def tensorboard_log(stats, tbd, step):
    num_sids = len(stats[args.eval_thresh[0]].keys())
    num_views = args.val_im_batch
    num_thresh = len(args.eval_thresh)
    ious = np.zeros((num_thresh, num_views, num_sids))
    for tx, th in enumerate(args.eval_thresh):
        for sx, sid in enumerate(stats[th].keys()):
            for nx, nix in enumerate(stats[th][sid].keys()):
                ious[tx, nx, sx] = np.array(stats[th][sid][nix]).mean()

    mean_ious = np.mean(ious, axis=2)
    for tx, th in enumerate(args.eval_thresh):
        for nx in range(num_views):
            tbd.log_scalar('iou_thresh_{:.1f}/views{:d}/val'.format(th, nx + 1),
                           mean_ious[tx, nx], step)


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
    vox_dir = SHAPENET_VOX[args.nvox]

    # Setup network
    net = model_vlsm(net, im_nets[args.im_net], grid_nets[args.grid_net],
                     conv_rnns[args.rnn])
    sess = tf.Session(config=get_session_config())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    coord = tf.train.Coordinator()

    # Init IoU
    iou = init_iou(net.im_batch, args.eval_thresh)

    # Init dataset
    dset = ShapeNet(
        im_dir=im_dir,
        split_file=args.val_split_file,
        vox_dir=vox_dir,
        rng_seed=1)
    mids = dset.get_smids('val')
    logging.info('Testing %d models', len(mids))
    items = ['shape_id', 'model_id', 'im', 'K', 'R', 'vol']
    dset.init_queue(
        mids,
        args.val_im_batch,
        items,
        coord,
        nepochs=1,
        qsize=32,
        nthreads=args.prefetch_threads)

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

            feed_dict = {net.K: batch_data['K'], net.Rcam: batch_data['R']}
            feed_dict[net.ims] = batch_data['im']

            pred = sess.run(net.prob_vox, feed_dict=feed_dict)
            batch_iou = eval_seq_iou(
                pred[:num_batch_items],
                batch_data['vol'][:num_batch_items],
                args.val_im_batch,
                thresh=args.eval_thresh)

            # Update iou dict
            iou = update_iou(batch_iou, iou)
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
    stats, iou_table = print_iou_stats(
        zip(deq_sids, deq_mids), iou, args.eval_thresh)

    return stats, iou_table


def parse_args():
    parser = argparse.ArgumentParser(description='Options for MVNet')
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--val_im_batch', type=int, default=4)
    parser.add_argument('--eval_thresh', type=float, nargs='+', default=[0.4])
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
