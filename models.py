import logging
import tensorflow as tf

from ops import (collapse_dims, conv2d, resize_conv2d, conv3d, convgru,
                 deconv2d, deconv3d, proj_slice, proj_splat, repeat_tensor,
                 separable_conv2d, tf_static_shape, uncollapse_dims)

#####################################
##### Image processing networks #####
#####################################


def im_unet(net, ims):
    net.im_net = {}
    bs, h, w, ch = ims.get_shape().as_list()
    with tf.variable_scope('ImNet_UNet'):
        conv1 = conv2d(
            'conv1', ims, 5, 32, act=None, norm=net.norm, mode=net.mode)
        net.im_net['conv1'] = conv1
        conv2 = conv2d('conv2', conv1, 3, 64, norm=net.norm, mode=net.mode)
        net.im_net['conv2'] = conv2
        conv3 = conv2d('conv3', conv2, 3, 128, norm=net.norm, mode=net.mode)
        net.im_net['conv3'] = conv3
        conv4 = conv2d('conv4', conv3, 3, 256, norm=net.norm, mode=net.mode)
        net.im_net['conv4'] = conv4
        _, fh, fw, ch = conv4.get_shape().as_list()
        deconv1 = deconv2d(
            'deconv1', conv4, 3, 128, norm=net.norm, mode=net.mode)
        net.im_net['deconv1'] = deconv1
        deconv1 = tf.concat([deconv1, conv3], axis=3)
        deconv2 = deconv2d(
            'deconv2', deconv1, 3, 64, norm=net.norm, mode=net.mode)
        net.im_net['deconv2'] = deconv2
        deconv2 = tf.concat([deconv2, conv2], axis=3)
        deconv3 = deconv2d(
            'deconv3', deconv2, 3, 32, norm=net.norm, mode=net.mode)
        net.im_net['deconv3'] = deconv3
        deconv3 = tf.concat([deconv3, conv1], axis=3)
        im_feats = deconv2d(
            'deconv4', deconv3, 5, 32, stride=1, norm=None, mode=net.mode)
        net.im_net['out'] = im_feats

    return im_feats


im_nets = {'unet': im_unet}

######################################
###### Grid processing networks ######
######################################


def grid_unet32(net, cost_vol):
    n, h, w, d, ch = cost_vol.get_shape().as_list()
    net.grid_net = {}
    with tf.variable_scope('Grid_Unet'):
        conv1 = conv3d(
            'conv1',
            cost_vol,
            4,
            32,
            activation=None,
            norm=net.norm,
            mode=net.mode)
        net.grid_net['conv1'] = conv1
        conv2 = conv3d('conv2', conv1, 4, 64, norm=net.norm, mode=net.mode)
        net.grid_net['conv2'] = conv2
        conv3 = conv3d('conv3', conv2, 4, 128, norm=net.norm, mode=net.mode)
        net.grid_net['conv3'] = conv3

        deconv1 = deconv3d(
            'deconv1', conv3, 4, 64, norm=net.norm, mode=net.mode)
        net.grid_net['deconv1'] = deconv1
        deconv1 = tf.concat([deconv1, conv2], axis=4)
        deconv2 = deconv3d(
            'deconv2', deconv1, 4, 32, norm=net.norm, mode=net.mode)
        net.grid_net['deconv2'] = deconv2
        deconv2 = tf.concat([deconv2, conv1], axis=4)
        deconv3 = deconv3d(
            'deconv3', deconv2, 4, 32, norm=net.norm, mode=net.mode)
        net.grid_net['deconv3'] = deconv3
        final_vol = deconv3d(
            'out', deconv3, 4, 1, stride=1, norm=None, mode=net.mode)
        net.grid_net['out'] = final_vol

    return final_vol


def grid_unet64(net, cost_vol):
    n, h, w, d, ch = cost_vol.get_shape().as_list()
    net.grid_net = {}
    with tf.variable_scope('Grid_Unet'):
        conv1 = conv3d(
            'conv1',
            cost_vol,
            4,
            32,
            activation=None,
            norm=net.norm,
            mode=net.mode)
        net.grid_net['conv1'] = conv1
        conv2 = conv3d('conv2', conv1, 4, 64, norm=net.norm, mode=net.mode)
        net.grid_net['conv2'] = conv2
        conv3 = conv3d('conv3', conv2, 4, 128, norm=net.norm, mode=net.mode)
        net.grid_net['conv3'] = conv3
        conv4 = conv3d('conv4', conv3, 4, 256, norm=net.norm, mode=net.mode)
        net.grid_net['conv4'] = conv4
        deconv1 = deconv3d(
            'deconv1', conv4, 4, 128, norm=net.norm, mode=net.mode)
        net.grid_net['deconv1'] = deconv1
        deconv1 = tf.concat([deconv1, conv3], axis=4)
        deconv2 = deconv3d(
            'deconv2', deconv1, 4, 64, norm=net.norm, mode=net.mode)
        net.grid_net['deconv2'] = deconv2
        deconv2 = tf.concat([deconv2, conv2], axis=4)
        deconv3 = deconv3d(
            'deconv3', deconv2, 4, 32, norm=net.norm, mode=net.mode)
        net.grid_net['deconv3'] = deconv3
        deconv3 = tf.concat([deconv3, conv1], axis=4)
        deconv4 = deconv3d(
            'deconv4', deconv3, 4, 32, norm=net.norm, mode=net.mode)
        net.grid_net['deconv4'] = deconv4
        final_vol = deconv3d(
            'out', deconv4, 4, 1, stride=1, norm=None, mode=net.mode)
        net.grid_net['out'] = final_vol

    return final_vol


grid_nets = {'unet32': grid_unet32, 'unet64': grid_unet64}

###################################
###### LSM graph definitions ######
###################################


def model_vlsm(net, im_net=im_unet, grid_net=grid_unet32, rnn=convgru):
    ''' Voxel LSTM model '''
    with tf.variable_scope('MVNet'):
        # Setup placeholders for image, extrinsics and intrinsics
        net.ims = tf.placeholder(tf.float32, net.im_tensor_shape, name='ims')
        net.K = tf.placeholder(tf.float32, net.K_tensor_shape, name='K')
        net.Rcam = tf.placeholder(tf.float32, net.R_tensor_shape, name='R')

        # Compute image features
        net.im_feats = im_net(net, collapse_dims(net.ims))

        # Unproject feature grid
        net.cost_grid = proj_splat(net, net.im_feats, net.K, net.Rcam)

        # Combine grids with LSTM/GRU
        net.pool_grid, _ = rnn(net.cost_grid)

        # 3D grid reasoning
        net.pool_grid = collapse_dims(net.pool_grid)
        net.pred_vox = grid_net(net, net.pool_grid)
        net.pred_vox = uncollapse_dims(net.pred_vox, net.batch_size,
                                       net.im_batch)
        net.prob_vox = tf.nn.sigmoid(net.pred_vox)
        return net


def model_dlsm(net,
               im_net=im_unet,
               grid_net=grid_unet32,
               rnn=convgru,
               ray_samples=64,
               proj_x=4,
               sepup=False,
               im_skip=True,
               proj_last=False):
    '''Depth LSTM model '''

    with tf.variable_scope('MVNet'):
        # Setup placeholders for im, depth, extrinsic and intrinsic matrices
        net.ims = tf.placeholder(tf.float32, net.im_tensor_shape, name='ims')
        net.K = tf.placeholder(tf.float32, net.K_tensor_shape, name='K')
        net.Rcam = tf.placeholder(tf.float32, net.R_tensor_shape, name='R')

        # Compute image features
        net.im_feats = im_net(net, collapse_dims(net.ims))

        # Unproject feature grid
        net.cost_grid = proj_splat(net, net.im_feats, net.K, net.Rcam)

        # Combine grids with LSTM/GRU
        net.pool_grid, _ = rnn(net.cost_grid)

        # Grid network
        net.pool_grid = collapse_dims(net.pool_grid)
        net.pred_vox = grid_net(net, net.pool_grid)
        net.proj_vox = uncollapse_dims(net.grid_net['deconv3'], net.batch_size,
                                       net.im_batch)

        # Projection
        proj_vox_in = (net.proj_vox
                       if not proj_last else net.proj_vox[:, -1:, ...])
        net.ray_slices, z_samples = proj_slice(
            net,
            proj_vox_in,
            net.K,
            net.Rcam,
            proj_size=net.im_h / proj_x,
            samples=ray_samples)

        bs, im_bs, ks, im_sz1, im_sz2, fdim, _ = tf_static_shape(
            net.ray_slices)
        net.depth_in = tf.reshape(net.ray_slices, [
            bs * im_bs * ks, im_sz1, im_sz2, fdim * ray_samples
        ])
        # Depth network
        if proj_x == 4:
            if not sepup:
                net.depth_out = depth_net_x4(net, net.depth_in, im_skip)
            else:
                net.depth_out = depth_net_x4_sepup(net, net.depth_in, im_skip)
        elif proj_x == 8:
            if not sepup:
                net.depth_out = depth_net_x8(net, net.depth_in, im_skip)
            else:
                net.depth_out = depth_net_x8_sepup(net, net.depth_in, im_skip)
        else:
            logger = logging.getLogger('mview3d.' + __name__)
            logger.error(
                'Unsupported subsample ratio for projection. Use {4, 8}')

        net.depth_out = tf.reshape(net.depth_out,
                                   [bs, im_bs, ks, net.im_h, net.im_w, 1])
        return net


##################################################
###### Depth prediction network definitions ######
##################################################


def depth_net_x4_sepup(net, in_, im_skip):
    def _skip_unet(d_f, im_f):
        ''' im_f: bs x im_bs x ... ; d_f: bs x t x im_bs ...'''
        with tf.variable_scope('Skip'):
            d_shape = tf_static_shape(d_f)
            im_shape = tf_static_shape(im_f)
            im_f = uncollapse_dims(im_f, net.batch_size, net.im_batch)
            im_rep = repeat_tensor(im_f, d_shape[0] / im_shape[0], rep_dim=1)
            im_rep = tf.reshape(im_rep, d_shape[:-1] + [im_shape[-1]])
            return tf.concat([im_rep, d_f], axis=-1)

    net.depth_net = {}
    with tf.variable_scope('DepthNet'):
        conv1 = separable_conv2d(
            'conv1', in_, 3, 1, 512, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv1'] = conv1  # 56x56
        conv2 = separable_conv2d(
            'conv2', conv1, 3, 1, 64, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv2'] = conv2  # 56x56

        conv2 = (_skip_unet(conv2, net.im_net['conv2']) if im_skip else conv2)
        deconv1 = resize_conv2d(
            'upconv1', conv2, 3, 32, norm=net.norm, mode=net.mode)  # 112 x 112
        net.depth_net['deconv1'] = deconv1
        deconv1 = (_skip_unet(deconv1, net.im_net['conv1'])
                   if im_skip else deconv1)
        deconv2 = resize_conv2d(
            'upconv2', deconv1, 3, 32, norm=net.norm,
            mode=net.mode)  # 224 x 224
        net.depth_net['deconv2'] = deconv2
        deconv3 = conv2d(
            'upconv3', deconv2, 3, 1, stride=1, norm=None,
            mode=net.mode)  # 224 x 224
        net.depth_net['deconv3'] = deconv3
        return net.depth_net['deconv3']


def depth_net_x8_sepup(net, in_, im_skip):
    def _skip_unet(d_f, im_f):
        ''' im_f: bs x im_bs x ... ; d_f: bs x t x im_bs ...'''
        with tf.variable_scope('Skip'):
            d_shape = tf_static_shape(d_f)
            im_shape = tf_static_shape(im_f)
            im_f = uncollapse_dims(im_f, net.batch_size, net.im_batch)
            im_rep = repeat_tensor(im_f, d_shape[0] / im_shape[0], rep_dim=1)
            im_rep = tf.reshape(im_rep, d_shape[:-1] + [im_shape[-1]])
            return tf.concat([im_rep, d_f], axis=-1)

    net.depth_net = {}
    with tf.variable_scope('DepthNet'):
        conv1 = separable_conv2d(
            'conv1', in_, 3, 1, 512, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv1'] = conv1  # 28x28
        conv2 = separable_conv2d(
            'conv2', conv1, 3, 1, 128, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv2'] = conv2  # 28x28

        conv2 = (_skip_unet(conv2, net.im_net['conv3']) if im_skip else conv2)
        deconv1 = resize_conv2d(
            'upconv1', conv2, 3, 64, norm=net.norm, mode=net.mode)  # 56x56
        net.depth_net['deconv1'] = deconv1
        deconv1 = (_skip_unet(deconv1, net.im_net['conv2'])
                   if im_skip else deconv1)
        deconv2 = resize_conv2d(
            'upconv2', deconv1, 3, 32, norm=net.norm, mode=net.mode)  # 112x112
        net.depth_net['deconv2'] = deconv2
        deconv2 = (_skip_unet(deconv2, net.im_net['conv1'])
                   if im_skip else deconv2)
        deconv3 = resize_conv2d(
            'upconv3', deconv2, 3, 32, norm=net.norm, mode=net.mode)  # 224x224
        net.depth_net['deconv3'] = deconv3
        deconv4 = conv2d(
            'upconv4', deconv3, 3, 1, stride=1, norm=None,
            mode=net.mode)  # 224x224
        net.depth_net['deconv4'] = deconv4
        return net.depth_net['deconv4']


def depth_net_x4(net, in_, im_skip):
    def _skip_unet(d_f, im_f):
        ''' im_f: bs x im_bs x ... ; d_f: bs x t x im_bs ...'''
        with tf.variable_scope('Skip'):
            d_shape = tf_static_shape(d_f)
            im_shape = tf_static_shape(im_f)
            im_f = uncollapse_dims(im_f, net.batch_size, net.im_batch)
            im_rep = repeat_tensor(im_f, d_shape[0] / im_shape[0], rep_dim=1)
            im_rep = tf.reshape(im_rep, d_shape[:-1] + [im_shape[-1]])
            return tf.concat([im_rep, d_f], axis=-1)

    net.depth_net = {}
    with tf.variable_scope('DepthNet'):
        conv1 = conv2d(
            'conv1', in_, 1, 512, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv1'] = conv1  # 56x56
        conv2 = conv2d(
            'conv2', conv1, 1, 128, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv2'] = conv2  # 56x56
        conv3 = conv2d(
            'conv3', conv2, 3, 64, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv3'] = conv3  # 56x56

        conv3 = (_skip_unet(conv3, net.im_net['conv2']) if im_skip else conv3)
        deconv1 = deconv2d(
            'deconv1', conv3, 3, 32, norm=net.norm, mode=net.mode)  # 112 x 112
        net.depth_net['deconv1'] = deconv1
        deconv1 = (_skip_unet(deconv1, net.im_net['conv1'])
                   if im_skip else deconv1)
        deconv2 = deconv2d(
            'deconv2', deconv1, 3, 32, norm=net.norm,
            mode=net.mode)  # 224 x 224
        net.depth_net['deconv2'] = deconv2
        deconv3 = conv2d(
            'deconv3', deconv2, 3, 1, stride=1, norm=None,
            mode=net.mode)  # 224 x 224
        net.depth_net['deconv3'] = deconv3
        return net.depth_net['deconv3']


def depth_net_x8(net, in_, im_skip):
    def _skip_unet(d_f, im_f):
        ''' im_f: bs x im_bs x ... ; d_f: bs x t x im_bs ...'''
        with tf.variable_scope('Skip'):
            d_shape = tf_static_shape(d_f)
            im_shape = tf_static_shape(im_f)
            im_f = uncollapse_dims(im_f, net.batch_size, net.im_batch)
            im_rep = repeat_tensor(im_f, d_shape[0] / im_shape[0], rep_dim=1)
            im_rep = tf.reshape(im_rep, d_shape[:-1] + [im_shape[-1]])
            return tf.concat([im_rep, d_f], axis=-1)

    net.depth_net = {}
    with tf.variable_scope('DepthNet'):
        conv1 = conv2d(
            'conv1', in_, 1, 512, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv1'] = conv1  # 28x28
        conv2 = conv2d(
            'conv2', conv1, 1, 128, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv2'] = conv2  # 28x28
        conv3 = conv2d(
            'conv3', conv2, 3, 128, stride=1, norm=net.norm, mode=net.mode)
        net.depth_net['conv3'] = conv3  # 28x28

        conv3 = (_skip_unet(conv3, net.im_net['conv3']) if im_skip else conv3)
        deconv1 = deconv2d(
            'deconv1', conv3, 3, 64, norm=net.norm, mode=net.mode)  # 56x56
        net.depth_net['deconv1'] = deconv1
        deconv1 = (_skip_unet(deconv1, net.im_net['conv2'])
                   if im_skip else deconv1)
        deconv2 = deconv2d(
            'deconv2', deconv1, 3, 32, norm=net.norm,
            mode=net.mode)  # 112 x 112
        net.depth_net['deconv2'] = deconv2
        deconv2 = (_skip_unet(deconv2, net.im_net['conv1'])
                   if im_skip else deconv2)
        deconv3 = deconv2d(
            'deconv3', deconv2, 3, 32, norm=net.norm,
            mode=net.mode)  # 224 x 224
        net.depth_net['deconv3'] = deconv3
        deconv4 = conv2d(
            'deconv4', deconv3, 3, 1, stride=1, norm=None,
            mode=net.mode)  # 224 x 224
        net.depth_net['deconv4'] = deconv4
        return net.depth_net['deconv4']
