import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

############################################
##### Utility functions for evaluation #####
############################################


def init_iou(im_batch, thresh):
    iou = dict()
    for ix in range(im_batch):
        iou[ix + 1] = dict()
        for k in thresh:
            iou[ix + 1][k] = []
    return iou


def update_iou(batch_iou, iou):
    for ix in iou.keys():
        for th in iou[ix].keys():
            iou[ix][th].extend(batch_iou[ix][th])

    return iou


def eval_seq_iou(pred, gt, im_batch, thresh=[0.1]):
    bs = gt.shape[0]
    gt = gt.astype(np.bool)
    iu = dict()
    for ix in range(im_batch):
        iu[ix + 1] = dict()
        for k in thresh:
            iu[ix + 1][k] = []

    for bx in range(im_batch):
        for th in thresh:
            for ix in range(bs):
                pred_t = (pred[ix][bx] > th).astype(np.bool)
                i = np.sum(np.logical_and(pred_t, gt[ix]))
                u = np.sum(np.logical_or(pred_t, gt[ix]))
                thiou = float(i) / u
                iu[bx + 1][th].append(thiou)

    return iu


def print_iou_stats(mids, iou, thresh, statistic='mean'):
    ''' mids: [(shape_id, model_id), ...]
        iou: {'#images': {'threshold': iou}}
        output: IoU Thresh: Shape_ids - mean iou'''

    def pline(s):
        return '\n' + '*' * 5 + ' ' + s + ' ' + '*' * 5

    shape_ids = np.unique([m[0] for m in mids])

    siou = dict()
    for th in thresh:
        siou[th] = dict()
        for sid in shape_ids:
            siou[th][sid] = dict()
            for ix in iou.keys():
                siou[th][sid][ix] = []

    for th in sorted(thresh):
        for mx, m in enumerate(mids):
            for ix in iou.keys():
                siou[th][m[0]][ix].append(iou[ix][th][mx])

    full_table = []
    for th in sorted(thresh):
        full_table.append(pline('IoU Thresh: {:.1f}'.format(th)))
        print_table = []
        for sid in shape_ids:
            print_table.append([sid])
            for ix in sorted(iou.keys()):
                if statistic == 'mean':
                    print_table[-1].append(
                        np.array(siou[th][sid][ix]).mean() * 100)
                elif statistic == 'median':
                    print_table[-1].append(
                        np.median(np.array(siou[th][sid][ix])) * 100)

        full_table.append(
            tabulate(print_table, headers=sorted(iou.keys()), floatfmt=".2f"))

    return siou, '\n'.join(full_table)


def vis_ims(ims, mask=None):
    if mask is not None:
        ims[np.logical_not(mask)] = None
    im_disp = np.reshape(ims, [-1] + list(ims.shape[2:]))
    im_d = np.concatenate([i for i in im_disp], axis=1)
    plt.imshow(np.uint8(im_d[..., 0] * 255))
    plt.axis('off')


def eval_l1_err(pred, gt, mask=None, vis=False):
    pred = pred[:, 0, ...]
    bs, im_batch = pred.shape[0], pred.shape[1]
    if mask is None:
        nanmask = (gt < np.max(gt))
    range_mask = np.logical_and(pred > 2.0 - np.sqrt(3) * 0.5,
                                pred < 2.0 + np.sqrt(3) * 0.5)
    mask = np.logical_and(nanmask, range_mask)

    if vis:
        plt.subplot(5, 1, 1)
        vis_ims(mask)
        plt.title("Eval Mask")
        plt.subplot(5, 1, 2)
        vis_ims(pred / 10.0, mask=mask)
        plt.title("Pred")
        plt.subplot(5, 1, 3)
        vis_ims(gt / 10.0, mask=nanmask)
        plt.title("Gt")
        plt.subplot(5, 1, 4)
        vis_ims(np.logical_xor(mask, nanmask))
        plt.title("Gt Mask - Mask")
        plt.subplot(5, 1, 5)
        vis_ims(np.abs(pred - gt) / 10.0, mask=mask)
        plt.title("Masked L1 error")
        plt.show()

    l1_err = np.abs(pred - gt)
    l1_err_masked = np.ma.array(l1_err, mask=np.logical_not(mask))
    batch_err = []
    for b in range(bs):
        tmp = np.zeros((im_batch, ))
        for imb in range(im_batch):
            tmp[imb] = np.ma.median(l1_err_masked[b, imb])
        batch_err.append(np.nanmean(tmp))
    return batch_err


def print_depth_stats(mids, err):
    shape_ids = np.unique([m[0] for m in mids])
    serr = dict()
    for sid in shape_ids:
        serr[sid] = []

    for ex, e in enumerate(err):
        serr[mids[ex][0]].append(e)

    table = []
    smean = []
    for s in serr:
        sm = np.nanmean(serr[s])
        table.append([s, sm])
        smean.append(sm)
    table.append(['Mean', np.nanmean(smean)])
    ptable = tabulate(table, headers=['SID', 'L1 error'], floatfmt=".4f")
    return smean, ptable
