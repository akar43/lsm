import json
import logging
import os.path as osp
from Queue import Empty, Queue
from threading import Thread, current_thread

import numpy as np

from config import SHAPENET_IM
from loader import read_camera, read_depth, read_im, read_quat, read_vol


def get_split(split_js='data/splits.json'):
    dir_path = osp.dirname(osp.realpath(__file__))
    with open(osp.join(dir_path, split_js), 'r') as f:
        js = json.load(f)

    return js


class ShapeNet(object):
    def __init__(self,
                 im_dir=SHAPENET_IM,
                 split_file='data/splits.json',
                 vox_dir=None,
                 shape_ids=None,
                 num_renders=20,
                 rng_seed=0):
        self.vox_dir = vox_dir
        self.im_dir = im_dir
        self.split_file = split_file
        self.splits_all = get_split(split_file)
        self.shape_ids = (self.splits_all.keys()
                          if shape_ids is None else shape_ids)
        self.splits = {k: self.splits_all[k] for k in self.shape_ids}

        self.shape_cls = [
            self.splits[x]['name'].split(',')[0] for x in self.shape_ids
        ]
        self.rng = rng_seed
        self.num_renders = num_renders
        self.load_func = {
            'im': self.get_im,
            'depth': self.get_depth,
            'K': self.get_K,
            'R': self.get_R,
            'quat': self.get_quat,
            'vol': self.get_vol,
            'shape_id': self.get_sid,
            'model_id': self.get_mid,
            'view_idx': self.get_view_idx
        }
        self.all_items = self.load_func.keys()

        self.logger = logging.getLogger('mview3d.' + __name__)
        np.random.seed(self.rng)

    def get_mids(self, sid):
        return self.splits[sid]

    def get_smids(self, split):
        smids = []
        for k, v in self.splits.iteritems():
            smids.extend([(k, m) for m in v[split]])
        smids = np.random.permutation(smids)
        return smids

    def get_sid(self, sid, mid, idx=None):
        return np.array([sid])

    def get_view_idx(self, sid, mid, idx):
        return idx

    def get_mid(self, sid, mid, idx=None):
        return np.array([mid])

    def get_K(self, sid, mid, idx):
        rand_idx = idx
        cams = []
        for ix in rand_idx:
            f = osp.join(self.im_dir, sid, mid, 'camera_{:d}.mat'.format(ix))
            cams.append(read_camera(f))
        camK = np.stack([c[0] for c in cams], axis=0)
        return camK

    def get_R(self, sid, mid, idx):
        rand_idx = idx
        cams = []
        for ix in rand_idx:
            f = osp.join(self.im_dir, sid, mid, 'camera_{:d}.mat'.format(ix))
            cams.append(read_camera(f))
        camR = np.stack([c[1] for c in cams], axis=0)
        return camR

    def get_quat(self, sid, mid, idx):
        rand_idx = idx
        cams = []
        for ix in rand_idx:
            f = osp.join(self.im_dir, sid, mid, 'camera_{:d}.mat'.format(ix))
            cams.append(read_quat(f))
        camq = np.stack(cams, axis=0)
        return camq

    def get_depth(self, sid, mid, idx):
        rand_idx = idx
        depths = []
        for ix in rand_idx:
            f = osp.join(self.im_dir, sid, mid, 'depth_{:d}.png'.format(ix))
            depths.append(read_depth(f))
        return np.stack(depths, axis=0)

    def get_im(self, sid, mid, idx):
        rand_idx = idx
        ims = []
        for ix in rand_idx:
            f = osp.join(self.im_dir, sid, mid, 'render_{:d}.png'.format(ix))
            ims.append(read_im(f))
        return np.stack(ims, axis=0)

    def get_vol(self, sid, mid, idx=None, tsdf=False):
        if self.vox_dir is None:
            self.logger.error('Voxel dir not defined')
        f = osp.join(self.vox_dir, sid, mid)
        return read_vol(f, tsdf)

    def fetch_data(self, smids, items, im_batch):
        with self.coord.stop_on_exception():
            while not self.coord.should_stop():
                data = {}
                try:
                    data_idx = self.queue_idx.get(timeout=0.5)
                except Empty:
                    self.logger.debug('Index queue empty - {:s}'.format(
                        current_thread().name))
                    continue

                view_idx = np.random.choice(
                    self.num_renders, size=(im_batch, ), replace=False)
                sid, mid = smids[data_idx]
                for i in items:
                    data[i] = self.load_func[i](sid, mid, view_idx)

                self.queue_data.put(data)
                if self.loop_data:
                    self.queue_idx.put(data_idx)

    def init_queue(self,
                   smids,
                   im_batch,
                   items,
                   coord,
                   nepochs=None,
                   qsize=32,
                   nthreads=4):
        self.coord = coord
        self.queue_data = Queue(maxsize=qsize)
        if nepochs is None:
            nepochs = 1
            self.loop_data = True
        else:
            self.loop_data = False
        self.total_items = nepochs * len(smids)
        self.queue_idx = Queue(maxsize=self.total_items)

        for nx in range(nepochs):
            for rx in range(len(smids)):
                self.queue_idx.put(rx)

        self.qthreads = []
        self.logger.info('Starting {:d} prefetch threads'.format(nthreads))
        for qx in range(nthreads):
            worker = Thread(
                target=self.fetch_data, args=(smids, items, im_batch))
            worker.start()
            self.coord.register_thread(worker)
            self.qthreads.append(worker)

    def close_queue(self, e=None):
        self.logger.debug('Closing queue')
        self.coord.request_stop(e)
        try:
            while True:
                self.queue_idx.get(block=False)
        except Empty:
            self.logger.debug('Emptied idx queue')

        try:
            while True:
                self.queue_data.get(block=False)
        except Empty:
            self.logger.debug("Emptied data queue")

    def next_batch(self, items, batch_size, timeout=0.5):
        data = []
        cnt = 0
        while cnt < batch_size:
            try:
                dt = self.queue_data.get(timeout=timeout)
                self.total_items -= 1
                data.append(dt)
            except Empty:
                self.logger.debug('Example queue empty')
                if self.total_items <= 0 and not self.loop_data:
                    # Exhausted all data
                    self.close_queue()
                    break
                else:
                    continue
            cnt += 1

        if len(data) == 0:
            return

        batch_data = {}
        for k in items:
            batch_data[k] = []
            for dt in data:
                batch_data[k].append(dt[k])
            batched = np.stack(batch_data[k])
            batch_data[k] = batched

        return batch_data

    def reset(self):
        np.random.seed(self.rng)
