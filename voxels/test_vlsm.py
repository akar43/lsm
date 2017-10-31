import argparse
import logging
import os.path as osp

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import SHAPENET_IM, SHAPENET_VOX
from loader import pad_batch
from models import grid_nets, im_nets, model_vlsm
from mvnet import MVNet
from ops import conv_rnns
from shapenet import ShapeNet
from utils import init_logging, process_args, get_session_config
from evaluate import eval_seq_iou, init_iou, print_iou_stats, update_iou


def run(args):
    net = MVNet(
        vmin=-0.5,
        vmax=0.5,
        vox_bs=args.test_batch_size,
        im_bs=args.test_im_batch,
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
    net.print_net()
    sess = tf.Session(config=get_session_config())
    saver = tf.train.Saver()
    saver.restore(sess, osp.join(args.log, args.ckpt))
    coord = tf.train.Coordinator()

    # Init IoU
    iou = init_iou(net.im_batch, args.eval_thresh)

    # Init dataset
    dset = ShapeNet(
        im_dir=im_dir,
        split_file=args.test_split_file,
        vox_dir=vox_dir,
        rng_seed=1)
    mids = dset.get_smids(args.split)
    print('Testing {:d} models'.format(len(mids)))
    items = ['shape_id', 'model_id', 'im', 'K', 'R', 'vol']
    dset.init_queue(
        mids,
        args.test_im_batch,
        items,
        coord,
        nepochs=1,
        qsize=32,
        nthreads=args.prefetch_threads)

    # Testing loop
    pbar = tqdm(desc='Testing', total=len(mids))
    deq_mids, deq_sids = [], []
    try:
        while not coord.should_stop():
            batch_data = dset.next_batch(items, net.batch_size)
            if batch_data is None:
                continue
            deq_sids.append(batch_data['shape_id'])
            deq_mids.append(batch_data['model_id'])
            num_batch_items = batch_data['K'].shape[0]
            batch_data = pad_batch(batch_data, args.test_batch_size)
            feed_dict = {net.K: batch_data['K'], net.Rcam: batch_data['R']}
            feed_dict[net.ims] = batch_data['im']

            pred = sess.run(net.prob_vox, feed_dict=feed_dict)
            batch_iou = eval_seq_iou(
                pred[:num_batch_items],
                batch_data['vol'][:num_batch_items],
                args.test_im_batch,
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
        logger.info('Testing completed')
        coord.join()

    deq_mids = np.concatenate(deq_mids, axis=0)[..., 0].tolist()
    deq_sids = np.concatenate(deq_sids, axis=0)[..., 0].tolist()

    # Print statistics and save to file
    stats, iou_table = print_iou_stats(
        zip(deq_sids, deq_mids), iou, args.eval_thresh)
    print(iou_table)
    if args.result_file is not None:
        print('Result written to: {:s}'.format(args.result_file))
        with open(args.result_file, 'w') as f:
            f.write(iou_table)


def parse_args():
    parser = argparse.ArgumentParser(description='Options for MVNet')
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--test_im_batch', type=int, default=4)
    parser.add_argument('--eval_thresh', type=float, nargs='+', default=[0.4])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--result_file', type=str, default=None)
    parser.add_argument('--loglevel', type=str, default='info')
    parser.add_argument(
        '--test_split_file', type=str, default='data/splits.json')
    parser.add_argument('--prefetch_threads', type=int, default=2)
    args = process_args(parser)
    return args


if __name__ == '__main__':
    args = parse_args()
    init_logging(args.loglevel)
    logger = logging.getLogger('mview3d.' + __name__)
    run(args)
