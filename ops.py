import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging

from convlstm import ConvGRUCell, ConvLSTMCell

logger = logging.getLogger('mview3d.' + __name__)


def get_bias(shape, name='bias'):
    return tf.get_variable(
        name, shape=shape, initializer=tf.constant_initializer(0.0))


def get_weights(shape, name='weights'):
    return tf.get_variable(
        name, shape=shape, initializer=slim.initializers.xavier_initializer())


def convgru(grid, kernel=[3, 3, 3], filters=32):
    bs, im_bs, h, w, d, ch = grid.get_shape().as_list()

    conv_gru = ConvGRUCell(
        shape=[h, w, d],
        initializer=slim.initializers.xavier_initializer(),
        kernel=kernel,
        filters=filters)
    seq_length = [im_bs for _ in range(bs)]
    outputs, states = tf.nn.dynamic_rnn(
        conv_gru,
        grid,
        parallel_iterations=64,
        sequence_length=seq_length,
        dtype=tf.float32,
        time_major=False)
    return outputs, states


def convlstm(grid, kernel=[3, 3, 3], filters=32):
    bs, im_bs, h, w, d, ch = grid.get_shape().as_list()

    conv_lstm = ConvLSTMCell(
        shape=[h, w, d],
        initializer=slim.initializers.xavier_initializer(),
        kernel=kernel,
        filters=filters)
    seq_length = [im_bs for _ in range(bs)]
    outputs, states = tf.nn.dynamic_rnn(
        conv_lstm,
        grid,
        parallel_iterations=64,
        sequence_length=seq_length,
        dtype=tf.float32,
        time_major=False)
    return outputs, states


conv_rnns = {'gru': convgru, 'lstm': convlstm}


def instance_norm(x):
    epsilon = 1e-5
    x_shape = x.get_shape().as_list()
    if len(x_shape) == 4:
        axis = [1, 2]
    elif len(x_shape) == 5:
        axis = [1, 2, 3]
    else:
        logger.error(
            'Instance norm not supported for tensor rank %d' % len(x_shape))
    with tf.variable_scope('InstanceNorm'):
        mean, var = tf.nn.moments(x, axis, keep_dims=True)
        beta = get_bias([x_shape[-1]])
        return tf.nn.batch_normalization(
            x, mean, var, offset=beta, scale=None, variance_epsilon=epsilon)


def deconv3d(name,
             X,
             fsize,
             ch,
             stride=2,
             norm=None,
             padding="SAME",
             activation=tf.nn.relu,
             mode="TRAIN"):
    bs, h, w, d, in_ch = tf_static_shape(X)
    filt_shape = [fsize, fsize, fsize, ch, in_ch]
    out_shape = [bs, h * stride, w * stride, d * stride, ch]
    stride = [1, stride, stride, stride, 1]
    with tf.variable_scope(name):
        if activation is not None:
            X = activation(X)

        params = get_weights(filt_shape)
        X = tf.nn.conv3d_transpose(X, params, out_shape, stride, padding)
        if norm is None:
            bias_dim = [filt_shape[-2]]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def conv3d(name,
           X,
           fsize,
           ch,
           stride=2,
           norm=None,
           padding="SAME",
           activation=tf.nn.relu,
           mode="TRAIN"):

    bs, h, w, d, in_ch = tf_static_shape(X)
    filt_shape = [fsize, fsize, fsize, in_ch, ch]
    stride = [1, stride, stride, stride, 1]
    with tf.variable_scope(name):
        if activation is not None:
            X = activation(X)
        params = get_weights(filt_shape)
        X = tf.nn.conv3d(X, params, stride, padding)
        if norm is None:
            bias_dim = [filt_shape[-1]]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def separable_conv2d(name,
                     X,
                     fsize,
                     ch_mult,
                     out_ch,
                     stride=2,
                     norm=None,
                     padding="SAME",
                     act=tf.nn.relu,
                     mode="TRAIN"):

    bs, h, w, in_ch = tf_static_shape(X)
    depth_filt_shape = [fsize, fsize, in_ch, ch_mult]
    point_filt_shape = [1, 1, in_ch * ch_mult, out_ch]
    stride = [1, stride, stride, 1]
    with tf.variable_scope(name):
        if act is not None:
            X = act(X)

        params_depth = get_weights(depth_filt_shape, name='depth_weights')
        params_pt = get_weights(point_filt_shape, name='pt_weights')

        X = tf.nn.depthwise_conv2d(X, params_depth, stride, padding)
        X = tf.nn.conv2d(X, params_pt, stride, padding)

        if norm is None:
            bias_dim = [out_ch]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def conv2d(name,
           X,
           fsize,
           ch,
           stride=2,
           norm=None,
           padding="SAME",
           act=tf.nn.relu,
           mode="TRAIN"):

    bs, h, w, in_ch = tf_static_shape(X)
    filt_shape = [fsize, fsize, in_ch, ch]
    stride = [1, stride, stride, 1]
    with tf.variable_scope(name):
        if act is not None:
            X = act(X)

        params = get_weights(filt_shape)
        X = tf.nn.conv2d(X, params, stride, padding)

        if norm is None:
            bias_dim = [filt_shape[-1]]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def resize_conv2d(name,
                  X,
                  fsize,
                  ch,
                  stride=2,
                  norm=None,
                  padding="SAME",
                  act=tf.nn.relu,
                  mode="TRAIN"):

    bs, h, w, in_ch = tf_static_shape(X)
    filt_shape = [fsize, fsize, in_ch, ch]
    new_h, new_w = h * stride, w * stride
    conv_stride = [1, 1, 1, 1]
    with tf.variable_scope(name):
        if act is not None:
            X = act(X)

        X = tf.image.resize_nearest_neighbor(X, [new_h, new_w])
        params = get_weights(filt_shape)
        X = tf.nn.conv2d(X, params, conv_stride, padding)

        if norm is None:
            bias_dim = [ch]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def deconv2d(name,
             X,
             fsize,
             ch,
             stride=2,
             norm=None,
             padding="SAME",
             act=tf.nn.relu,
             mode="TRAIN"):

    bs, h, w, in_ch = tf_static_shape(X)
    filt_shape = [fsize, fsize, ch, in_ch]
    out_shape = [bs, h * stride, w * stride, ch]
    stride = [1, stride, stride, 1]
    with tf.variable_scope(name):
        if act is not None:
            X = act(X)

        params = get_weights(filt_shape)
        X = tf.nn.conv2d_transpose(X, params, out_shape, stride, padding)

        if norm is None:
            bias_dim = [filt_shape[-2]]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def residual(x, channels=32, norm="IN", scope='res'):
    with tf.variable_scope(scope):
        res_x = conv2d(
            scope + '_1', x, 3, channels, stride=1, norm=norm, act=None)
        res_x = conv2d(scope + '_2', res_x, 3, channels, stride=1, norm=norm)
        x_skip = conv2d(scope + '_s', x, 1, channels, stride=1, norm=norm)
        return res_x + x_skip


def fully_connected(name, X, dim, activation=tf.nn.relu):
    bs = X.get_shape().as_list()[0]
    with tf.variable_scope(name):
        if activation is not None:
            X = activation(X)
        X = tf.reshape(X, [bs, -1])
        wshape = (X.get_shape().as_list()[-1], dim)
        params = get_weights(wshape)
        X = tf.matmul(X, params)
        X = tf.nn.bias_add(X, get_bias(dim))
    return X


def nearest3(grid, idx, clip=False):
    with tf.variable_scope('NearestInterp'):
        _, h, w, d, f = grid.get_shape().as_list()
        x, y, z = idx[:, 1], idx[:, 2], idx[:, 3]
        g_val = tf.gather_nd(grid, tf.cast(tf.round(idx), 'int32'))
        if clip:
            x_inv = tf.logical_or(x < 0, x > h - 1)
            y_inv = tf.logical_or(y < 0, y > w - 1)
            z_inv = tf.logical_or(z < 0, x > d - 1)
            valid_idx = 1 - \
                tf.to_float(tf.logical_or(tf.logical_or(x_inv, y_inv), z_inv))
            g_val = g_val * valid_idx[tf.newaxis, ...]
        return g_val


def proj_slice(net,
               grid,
               K,
               R,
               proj_size=224,
               samples=64,
               min_z=1.0,
               max_z=3.0):
    '''grid = nv grids, R = nv x nr rotation matrices, '''
    ''' R = (bs, im, 3, 4), K = (bs, im, 3, 3), grid = (bs, im, h, w, d, ch)'''
    rsz_factor = float(proj_size) / net.im_h
    K = K * rsz_factor
    K_shape = tf_static_shape(K)
    bs, im_bs, h, w, d, ch = tf_static_shape(grid)
    npix = proj_size**2
    with tf.variable_scope('ProjSlice'):
        # Setup dimensions
        with tf.name_scope('PixelCenters'):
            # Setup image grids to unproject along rays
            im_range = tf.range(0.5, proj_size, 1)
            im_grid = tf.stack(tf.meshgrid(im_range, im_range))
            rs_grid = tf.reshape(im_grid, [2, -1])
            # Append rsz_factor to ensure that
            rs_grid = tf.concat(
                [rs_grid, tf.ones((1, npix)) * rsz_factor], axis=0)
            rs_grid = tf.reshape(rs_grid, [1, 1, 3, npix])
            rs_grid = tf.tile(rs_grid, [K_shape[0], K_shape[1], 1, 1])

        with tf.name_scope('Im2Cam'):
            # Compute Xc - points in camera frame
            Xc = tf.matrix_triangular_solve(
                K, rs_grid, lower=False, name='KinvX')

            # Define z values of samples along ray
            z_samples = tf.linspace(min_z, max_z, samples)

            # Transform Xc to Xw using transpose of rotation matrix
            Xc = repeat_tensor(Xc, samples, rep_dim=2)
            Xc = Xc * z_samples[tf.newaxis, tf.newaxis, :, tf.newaxis,
                                tf.newaxis]
            Xc = tf.concat(
                [Xc, tf.ones([K_shape[0], K_shape[1], samples, 1, npix])],
                axis=-2)

        with tf.name_scope('Cam2World'):
            # Construct [R^{T}|-R^{T}t]
            Rt = tf.matrix_transpose(R[:, :, :, :3])
            tr = tf.expand_dims(R[:, :, :, 3], axis=-1)
            R_c2w = tf.concat([Rt, -tf.matmul(Rt, tr)], axis=3)
            R_c2w = repeat_tensor(R_c2w, samples, rep_dim=2)
            Xw = tf.matmul(R_c2w, Xc)

            # Transform world points to grid locations to sample from
            Xw = ((Xw - net.vmin) / (net.vmax - net.vmin)) * net.nvox
            # bs, K_shape[1], samples, npix, 3
            Xw = tf.transpose(Xw, [0, 1, 2, 4, 3])
            Xw = repeat_tensor(Xw, im_bs, rep_dim=1)

        with tf.name_scope('Interp'):
            sample_grid = collapse_dims(grid)
            sample_locs = collapse_dims(Xw)
            lshape = tf_static_shape(sample_locs)
            vox_idx = tf.range(lshape[0])
            vox_idx = repeat_tensor(vox_idx, lshape[1], rep_dim=1)
            vox_idx = tf.reshape(vox_idx, [-1, 1])
            vox_idx = repeat_tensor(vox_idx, samples * npix, rep_dim=1)
            vox_idx = tf.reshape(vox_idx, [-1, 1])
            sample_idx = tf.concat(
                [tf.to_float(vox_idx),
                 tf.reshape(sample_locs, [-1, 3])],
                axis=1)
            g_val = nearest3(sample_grid, sample_idx)
            g_val = tf.reshape(g_val, [
                bs, im_bs, K_shape[1], samples, proj_size, proj_size, -1
            ])
            ray_slices = tf.transpose(g_val, [0, 1, 2, 4, 5, 6, 3])

        return ray_slices, z_samples


def proj_splat(net, feats, K, Rcam):
    KRcam = tf.matmul(K, Rcam)
    with tf.variable_scope('ProjSplat'):
        nR, fh, fw, fdim = tf_static_shape(feats)
        rsz_h = float(fh) / net.im_h
        rsz_w = float(fw) / net.im_w

        # Create voxel grid
        with tf.name_scope('GridCenters'):
            grid_range = tf.range(net.vmin + net.vsize / 2.0, net.vmax,
                                  net.vsize)
            net.grid = tf.stack(
                tf.meshgrid(grid_range, grid_range, grid_range))
            net.rs_grid = tf.reshape(net.grid, [3, -1])
            nV = tf_static_shape(net.rs_grid)[1]
            net.rs_grid = tf.concat([net.rs_grid, tf.ones([1, nV])], axis=0)

        # Project grid
        with tf.name_scope('World2Cam'):
            im_p = tf.matmul(tf.reshape(KRcam, [-1, 4]), net.rs_grid)
            im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]
            im_x = (im_x / im_z) * rsz_w
            im_y = (im_y / im_z) * rsz_h
            net.im_p, net.im_x, net.im_y, net.im_z = im_p, im_x, im_y, im_z

        # Bilinear interpolation
        with tf.name_scope('BilinearInterp'):
            im_x = tf.clip_by_value(im_x, 0, fw - 1)
            im_y = tf.clip_by_value(im_y, 0, fh - 1)
            im_x0 = tf.cast(tf.floor(im_x), 'int32')
            im_x1 = im_x0 + 1
            im_y0 = tf.cast(tf.floor(im_y), 'int32')
            im_y1 = im_y0 + 1
            im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
            im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)
            
            ind_grid = tf.range(0, nR)
            ind_grid = tf.expand_dims(ind_grid, 1)
            im_ind = tf.tile(ind_grid, [1, nV])

            def _get_gather_inds(x, y):
                return tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])

            # Gather  values
            Ia = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y0))
            Ib = tf.gather_nd(feats, _get_gather_inds(im_x0, im_y1))
            Ic = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y0))
            Id = tf.gather_nd(feats, _get_gather_inds(im_x1, im_y1))

            # Calculate bilinear weights
            wa = (im_x1_f - im_x) * (im_y1_f - im_y)
            wb = (im_x1_f - im_x) * (im_y - im_y0_f)
            wc = (im_x - im_x0_f) * (im_y1_f - im_y)
            wd = (im_x - im_x0_f) * (im_y - im_y0_f)
            wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
            wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
            net.wa, net.wb, net.wc, net.wd = wa, wb, wc, wd
            net.Ia, net.Ib, net.Ic, net.Id = Ia, Ib, Ic, Id
            Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

        with tf.name_scope('AppendDepth'):
            # Concatenate depth value along ray to feature
            Ibilin = tf.concat(
                [Ibilin, tf.reshape(im_z, [nV * nR, 1])], axis=1)
            fdim = Ibilin.get_shape().as_list()[-1]
            net.Ibilin = tf.reshape(Ibilin, [
                net.batch_size, net.im_batch, net.nvox, net.nvox, net.nvox,
                fdim
            ])
            net.Ibilin = tf.transpose(net.Ibilin, [0, 1, 3, 2, 4, 5])
    return net.Ibilin


def loss_l1(pred, gt):
    return tf.losses.absolute_difference(gt, pred, scope='loss_l1')


def loss_ce(pred, gt_vox):
    with tf.variable_scope('loss_ce'):
        pred = tf.expand_dims(tf.reshape(pred, [-1]), axis=1)
        gt_vox = tf.expand_dims(tf.reshape(gt_vox, [-1]), axis=1)
        return tf.losses.sigmoid_cross_entropy(gt_vox, pred)


def concat_pool(feats):
    batch_size = feats.get_shape().as_list()[0]
    nvox = feats.get_shape().as_list()[2]
    with tf.variable_scope('concat_pool'):
        feats = tf.transpose(feats, [0, 5, 1, 2, 3, 4])
        feats = tf.reshape(feats, [batch_size, -1, nvox, nvox, nvox])
        feats = tf.transpose(feats, [0, 2, 3, 4, 1])
    return feats


def form_image_grid(input_tensor, grid_shape, image_shape, num_channels):
    """Arrange a minibatch of images into a grid to form a single image.
    Args:
      input_tensor: Tensor. Minibatch of images to format, either 4D
          ([batch size, height, width, num_channels]) or flattened
          ([batch size, height * width * num_channels]).
      grid_shape: Sequence of int. The shape of the image grid,
          formatted as [grid_height, grid_width].
      image_shape: Sequence of int. The shape of a single image,
          formatted as [image_height, image_width].
      num_channels: int. The number of channels in an image.
    Returns:
      Tensor representing a single image in which the input images have been
      arranged into a grid.
    Raises:
      ValueError: The grid shape and minibatch size don't match, or the image
          shape and number of channels are incompatible with the input tensor.
    """
    if grid_shape[0] * grid_shape[1] != int(input_tensor.get_shape()[0]):
        raise ValueError('Grid shape incompatible with minibatch size.')
    if len(input_tensor.get_shape()) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.get_shape()[1]) != num_features:
            raise ValueError(
                'Image shape and number of channels incompatible with '
                'input tensor.')
    elif len(input_tensor.get_shape()) == 4:
        if (int(input_tensor.get_shape()[1]) != image_shape[0] or
                int(input_tensor.get_shape()[2]) != image_shape[1] or
                int(input_tensor.get_shape()[3]) != num_channels):
            raise ValueError(
                'Image shape and number of channels incompatible with'
                'input tensor.')
    else:
        raise ValueError('Unrecognized input tensor format.')
    height, width = grid_shape[0] * \
        image_shape[0], grid_shape[1] * image_shape[1]
    input_tensor = tf.reshape(input_tensor,
                              grid_shape + image_shape + [num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
    input_tensor = tf.reshape(
        input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
    input_tensor = tf.reshape(input_tensor, [1, height, width, num_channels])
    return input_tensor


def voxel_views(in_vox, gh, gw, tsdf=False, pad=4, scope='voxel_views'):
    def vis_tsdf(tv):
        return 1.0 - tf.abs(tv) * 5.0

    with tf.variable_scope(scope):
        _, h, w, d, ch = in_vox.get_shape().as_list()
        if not tsdf:
            x_view = tf.reduce_max(in_vox, axis=1)
            y_view = tf.reduce_max(in_vox, axis=2)
            z_view = tf.reduce_max(in_vox, axis=3)
        else:
            x_view = vis_tsdf(in_vox[:, h / 2, :, :, :])
            y_view = vis_tsdf(in_vox[:, :, w / 2, :, :])
            z_view = vis_tsdf(in_vox[:, :, :, d / 2, :])

        pad = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
        x_view = tf.pad(x_view, pad)
        y_view = tf.pad(y_view, pad)
        z_view = tf.pad(z_view, pad)
        image_shape = x_view.get_shape().as_list()[1:3]
        grid_shape = [gh, gw]
        color_ch = 1
        x_view = form_image_grid(x_view, grid_shape, image_shape, color_ch)
        y_view = form_image_grid(y_view, grid_shape, image_shape, color_ch)
        z_view = form_image_grid(z_view, grid_shape, image_shape, color_ch)
        views = [
            tf.cast(x_view * 255, tf.uint8),
            tf.cast(y_view * 255, tf.uint8),
            tf.cast(z_view * 255, tf.uint8)
        ]
        return views


def im_views(ims, gh, gw, scope='im_views'):
    with tf.variable_scope(scope):
        _, _, h, w, ch = tf_static_shape(ims)
        im_grid = form_image_grid(collapse_dims(ims), [gh, gw], [h, w], ch)
        return tf.cast(im_grid * 255, tf.uint8)


def voxel_sum(net, tsdf=False):
    vox_sum = []
    pred_views = voxel_views(
        collapse_dims(net.pred_vox),
        net.batch_size,
        net.im_batch,
        tsdf,
        scope='vox_pred')
    gt_views = voxel_views(net.gt_vox, net.batch_size, 1, tsdf, scope='vox_gt')
    caxis = 2

    with tf.name_scope('voxel_vis'):
        x_view = tf.concat([pred_views[0], gt_views[0]], axis=caxis)
        vox_sum.append(tf.summary.image('x_view', x_view))
        y_view = tf.concat([pred_views[1], gt_views[1]], axis=caxis)
        vox_sum.append(tf.summary.image('y_view', y_view))
        z_view = tf.concat([pred_views[2], gt_views[2]], axis=caxis)
        vox_sum.append(tf.summary.image('z_view', z_view))
        return tf.summary.merge(vox_sum)


def image_sum(im_tensor, nh, nw, tag='views'):
    return tf.summary.image(tag + '_sum', im_views(im_tensor, nh, nw, tag))


def vis_depth(d, min_d=1, max_d=3, sc=10):
    with tf.name_scope('vis_depth'):
        d_alpha = tf.to_float(tf.logical_and(d < max_d, d > min_d))
        d_v = d / sc * max_d
        d_v = tf.concat([d_v, d_v, d_v, d_alpha], axis=-1)
        return d_v


def depth_sum(depth_tensor, nh, nw, tag='depth_views'):
    return tf.summary.image(tag + '_sum',
                            im_views(vis_depth(depth_tensor), nh, nw, tag))


def repeat_tensor(T, nrep, rep_dim=1):
    repT = tf.expand_dims(T, rep_dim)
    tile_dim = [1] * len(tf_static_shape(repT))
    tile_dim[rep_dim] = nrep
    repT = tf.tile(repT, tile_dim)
    return repT


def collapse_dims(T):
    shape = tf_static_shape(T)
    return tf.reshape(T, [-1] + shape[2:])


def uncollapse_dims(T, s1, s2):
    shape = tf_static_shape(T)
    return tf.reshape(T, [s1, s2] + shape[1:])


def tf_static_shape(T):
    return T.get_shape().as_list()
