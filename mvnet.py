from utils import pretty_line


class MVNet(object):
    def __init__(self,
                 vmin,
                 vmax,
                 vox_bs,
                 im_bs,
                 grid_size,
                 im_h,
                 im_w,
                 mode="TRAIN",
                 norm="IN"):
        self.batch_size = vox_bs
        self.im_batch = im_bs
        self.nvox = grid_size
        self.im_h = im_h
        self.im_w = im_w
        self.vmin = vmin
        self.vmax = vmax
        self.vsize = float(self.vmax - self.vmin) / self.nvox
        self.mode = mode
        self.norm = norm

    @property
    def vox_tensor_shape(self):
        return [self.batch_size, self.nvox, self.nvox, self.nvox, 1]

    @property
    def vfp_vox_tensor_shape(self):
        return [
            self.batch_size, self.im_batch, self.nvox, self.nvox, self.nvox, 1
        ]

    @property
    def im_tensor_shape(self):
        return [self.batch_size, self.im_batch, self.im_h, self.im_w, 3]

    @property
    def depth_tensor_shape(self):
        return [self.batch_size, self.im_batch, self.im_h, self.im_w, 1]

    @property
    def K_tensor_shape(self):
        return [self.batch_size, self.im_batch, 3, 3]

    @property
    def R_tensor_shape(self):
        return [self.batch_size, self.im_batch, 3, 4]

    @property
    def quat_tensor_shape(self):
        return [self.batch_size, self.im_batch, 4]

    @property
    def total_ims_per_batch(self):
        return self.batch_size * self.im_batch

    def print_net(self):
        if hasattr(self, 'im_net'):
            print '\n'
            pretty_line('Image Encoder')
            for k, v in sorted(self.im_net.iteritems()):
                print k + '\t' + str(v.get_shape().as_list())

        if hasattr(self, 'grid_net'):
            print '\n'
            pretty_line('Grid Net')
            for k, v in sorted(self.grid_net.iteritems()):
                print k + '\t' + str(v.get_shape().as_list())

        if hasattr(self, 'depth_net'):
            print '\n'
            pretty_line('Depth Net')
            for k, v in sorted(self.depth_net.iteritems()):
                print k + '\t' + str(v.get_shape().as_list())

        if hasattr(self, 'encoder'):
            print '\n'
            pretty_line('Encoder')
            for k, v in sorted(self.encoder.iteritems()):
                print k + '\t' + str(v.get_shape().as_list())

        if hasattr(self, 'decoder'):
            print '\n'
            pretty_line('Decoder')
            for k, v in sorted(self.decoder.iteritems()):
                print k + '\t' + str(v.get_shape().as_list())

        return
