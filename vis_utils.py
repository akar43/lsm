import os

import numpy as np
from IPython.display import IFrame
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.ndimage.filters import median_filter

from config import BASE_DIR, SHAPENET_IM
from utils import mkdir_p
from uuid import uuid4
with open(os.path.join(BASE_DIR, 'pyntcloud.js'), 'r') as f:
    TEMPLATE_POINTS = f.read()


def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array, norm=False)[:, :-1]


def plot_points(xyz,
                colors=None,
                size=0.1,
                axis=False,
                title=None,
                html_out=None):
    positions = xyz.reshape(-1).tolist()
    mkdir_p('vis')
    if html_out is None:
        html_out = os.path.join('vis', 'pts{:s}.html'.format(uuid4()))
    if title is None:
        title = "PointCloud"
    camera_position = xyz.max(0) + abs(xyz.max(0))

    look = xyz.mean(0)

    if colors is None:
        colors = [1, 0.5, 0] * len(positions)

    elif len(colors.shape) > 1:
        colors = colors.reshape(-1).tolist()

    if axis:
        axis_size = xyz.ptp() * 1.5
    else:
        axis_size = 0

    with open(html_out, "w") as html:
        html.write(
            TEMPLATE_POINTS.format(
                title=title,
                camera_x=camera_position[0],
                camera_y=camera_position[1],
                camera_z=camera_position[2],
                look_x=look[0],
                look_y=look[1],
                look_z=look[2],
                positions=positions,
                colors=colors,
                points_size=size,
                axis_size=axis_size))

    return IFrame(html_out, width=1024, height=768)


def image_grid(ims, mask=None):
    if mask is not None:
        ims[np.logical_not(mask)] = None
    gh, gw, h, w, ch = ims.shape
    disp_im = np.zeros([gh * h, gw * w, ch])
    for y in range(gh):
        for x in range(gw):
            disp_im[y * h:(y + 1) * h, x * w:(x + 1) * w, :] = ims[y][x]
    return disp_im


def voxel_grid(voxels, thresh=0.4, cmap='viridis'):
    gh, gw, h, w, d, ch = voxels.shape
    all_pts, all_clr = [], []
    for bx in range(gh):
        for ix in range(gw):
            pts, clr = voxels2pts(voxels[bx][ix], cmap=cmap)
            pts[:, 0] += ix * w
            pts[:, 1] += bx * h
            all_pts.append(pts)
            all_clr.append(clr)
        all_pts.append(pts)
        all_clr.append(clr)
    vis_pts, vis_clr = np.concatenate(
        all_pts, axis=0), np.concatenate(
            all_clr, axis=0)
    return vis_pts, vis_clr


def voxels2pts(voxels, thresh=0.4, cmap="Oranges"):
    if voxels.ndim == 4:
        fvox = voxels[..., 0]
    elif voxels.ndim == 3:
        fvox = voxels
    else:
        print('Invalid number of dimension in voxel grid')
        return
    vox = (fvox > thresh).astype(np.int)
    points = np.argwhere(vox > 0)
    colors = array_to_color(fvox[vox > 0], cmap=cmap)
    return points, colors


def voxel2mesh(voxels):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                  [1, 0, 1], [1, 1, 0], [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6],
                  [0, 6, 4], [0, 5, 1], [0, 4, 5], [6, 7, 5], [6, 5, 4],
                  [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    l, m, n = voxels.shape

    scale = 1.0 / voxels.shape[0]
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0
    for i in range(l):
        for j in range(m):
            for k in range(n):
                # If there is a non-empty voxel
                if voxels[i, j, k] > 0:
                    verts.extend(
                        scale *
                        (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
                    faces.extend(cube_faces + curr_vert)
                    curr_vert += len(cube_verts)

    verts = np.array(verts) - 0.5
    return verts, np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred):
    verts, faces = voxel2mesh(pred)
    write_obj(filename, verts, faces)


def unproject_depth(d_im, K, R, im=None, dmin=1.0, dmax=3.0):
    px, py, f = K[0, 2], K[1, 2], K[0, 0]
    size = d_im.shape
    x, y = np.meshgrid(range(size[0]), range(size[1]))
    x, y = (x - px) * d_im / f, (y - py) * d_im / f
    xyz = np.stack([x, y, d_im], axis=0)
    xyz = np.reshape(xyz, [3, -1])
    mask = np.logical_and(xyz[-1, :] < dmax, xyz[-1, :] > dmin)
    xyz = xyz[:, mask]
    clr = None
    if im is not None:
        im = np.transpose(im, [2, 0, 1])
        im = np.reshape(im, [3, -1])
        clr = im[:, mask]

    tr = -np.matmul(R[:3, :3].T, R[:, 3][..., np.newaxis])
    Rt = np.concatenate([R[:3, :3].T, tr], axis=1)
    Xw = np.matmul(Rt,
                   np.concatenate([xyz, np.ones((1, xyz.shape[1]))], axis=0))
    return np.transpose(Xw), np.transpose(clr)


def depth2mesh(classId,
               shapeId,
               dmap,
               im_idx,
               dmin=1.15,
               dmax=2.85,
               discThresh=0.035,
               smooth=True,
               obj=None):
    shapeNetFolder = os.path.join(SHAPENET_IM, classId, shapeId)
    camera_f = os.path.join(shapeNetFolder, 'camera_{:d}.mat'.format(im_idx))
    mat = loadmat(camera_f)
    K = mat['K']
    px, py, f = K[0, 2], K[1, 2], K[0, 0]
    R = mat['extrinsic'][:3, :]
    tr = -np.matmul(R[:3, :3].T, R[:, 3][..., np.newaxis])
    Rt = np.concatenate([R[:3, :3].T, tr], axis=1)
    if obj is None:
        out_f = 'depth/mesh_{}_{}_{}.obj'.format(classId, shapeId, im_idx)
    else:
        out_f = obj

    depthMap = dmap
    if smooth:
        depthMap = median_filter(depthMap, (3, 3))

    h, w = depthMap.shape
    allPoints = np.ndarray(shape=(h, w, 5))

    with open(out_f, 'w') as obj:
        ind = 1
        for y in range(0, h):
            for x in range(0, w):
                d_im = depthMap[y, x]
                allPoints[y, x, 3] = depthMap[y, x]
                if (d_im < dmax and d_im > dmin):
                    x_c, y_c = (x - px) * d_im / f, (y - py) * d_im / f
                    upointLocal = np.array([[x_c], [y_c], [d_im], [1]])
                    upoint = np.matmul(Rt, upointLocal)
                    allPoints[y, x, :3] = upoint[:3, 0]
                    allPoints[y, x, 4] = ind
                    ind = ind + 1
                    obj.write('v {} {} {}\n'.format(upoint[0, 0], upoint[1, 0],
                                                    upoint[2, 0]))

        for y in range(0, h - 1):
            for x in range(0, w - 1):
                v = allPoints[y, x, 4]
                vd = allPoints[y, x, 3]
                vx = allPoints[y, x + 1, 4]
                vxd = allPoints[y, x + 1, 3]
                vy = allPoints[y + 1, x, 4]
                vyd = allPoints[y + 1, x, 3]
                vxy = allPoints[y + 1, x + 1, 4]
                vxyd = allPoints[y + 1, x + 1, 3]

                t1_minD = min(vd, vxd, vyd)
                t1_maxD = max(vd, vxd, vyd)

                if (t1_minD > dmin and t1_maxD < dmax and
                        t1_maxD - t1_minD < discThresh):
                    obj.write(
                        'f {:d} {:d} {:d}\n'.format(int(vy), int(vx), int(v)))

                t2_minD = min(vxyd, vxd, vyd)
                t2_maxD = max(vxyd, vxd, vyd)

                if (t2_minD > dmin and t2_maxD < dmax and
                        t2_maxD - t2_minD < discThresh):
                    try:
                        obj.write('f {:d} {:d} {:d}\n'.format(
                            int(vxy), int(vx), int(vy)))
                    except:
                        print 'Error', vxy, vx, vy
                        return
    return out_f
