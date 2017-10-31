import errno
import json
import logging
import logging.config
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
from scipy.linalg import logm, norm

#####################################
##### General utility functions #####
#####################################


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=False):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.__init__()


def mkdir_p(path):
    """Utility function emulating mkdir -p."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def pretty_line(s):
    print '*' * 5 + ' ' + s + ' ' + '*' * 5


def write_args(args, jsonfile):
    dump = dict()
    for arg in vars(args):
        dump[arg] = getattr(args, arg)

    with open(jsonfile, 'w') as dumpfile:
        json.dump(dump, dumpfile, indent=4, sort_keys=True)


def init_logging(level="INFO"):
    logging.basicConfig(
        format='%(asctime)s:%(module)s - %(name)s: %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('mview3d')
    numlevel = getattr(logging, level.upper(), None)
    if not isinstance(numlevel, int):
        raise ValueError('Invalid log level: %s' % level)
    logger.setLevel(numlevel)
    return logger


def get_session_config(memfrac=1.0):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = memfrac
    return config


###################################################################
##### Utility functions for processing command line arguments #####
###################################################################


def process_args(parser, js=None):
    args = parser.parse_args()
    argsdict = vars(args)

    # First search for argsjs, then args.log else None
    if js is None:
        if 'argsjs' in argsdict:
            js = args.argsjs
        elif 'log' in argsdict:
            if args.log is not None:
                js = os.path.join(args.log, 'args.json')
            else:
                js = None
        else:
            js = None

    def get_cmd_args():
        cmd_arg = {}
        for arg in sys.argv[1:]:
            for a in parser._actions:
                for o in a.option_strings:
                    if arg.startswith(o):
                        cmd_arg[a.dest] = argsdict[a.dest]
        return cmd_arg

    if js is not None:
        if not os.path.exists(js):
            print('Error: Specified args json file not found at {}'.format(js))
            print('Returning default args')
            return args

        with open(js, 'r') as f:
            js = json.load(f)

        # Get values specified on cmd line
        cmd_args = get_cmd_args()
        # Update values from json file
        argsdict.update(js)
        # Keep values specified in command line args
        argsdict.update(cmd_args)
        # Return namespace object
        args = Bunch(argsdict)

    return args


##################################################
##### Utility function for rotation matrices #####
##################################################


def quat2rot(q):
    '''q = [w, x, y, z]
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion'''
    eps = 1e-5
    w, x, y, z = q
    n = np.linalg.norm(q)
    s = (0 if n < eps else 2.0 / n)
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    R = np.array([[1 - (yy + zz), xy - wz,
                   xz + wy], [xy + wz, 1 - (xx + zz), yz - wx],
                  [xz - wy, yz + wx, 1 - (xx + yy)]])
    return R


def rot2quat(M):
    if M.shape[0] < 4 or M.shape[1] < 4:
        newM = np.zeros((4, 4))
        newM[:3, :3] = M[:3, :3]
        newM[3, 3] = 1
        M = newM

    q = np.empty((4, ))
    t = np.trace(M)
    if t > M[3, 3]:
        q[0] = t
        q[3] = M[1, 0] - M[0, 1]
        q[2] = M[0, 2] - M[2, 0]
        q[1] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def euler_to_rot(theta):

    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def az_el_to_rot(az, el):
    corr_mat = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    inv_corr_mat = np.linalg.inv(corr_mat)

    def R_x(theta):
        return np.array([[1, 0, 0], [0, math.cos(theta),
                                     math.sin(theta)],
                         [0, -math.sin(theta),
                          math.cos(theta)]])

    def R_y(theta):
        return np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0],
                         [math.sin(theta), 0,
                          math.cos(theta)]])

    def R_z(theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    Rmat = np.matmul(R_x(-el * math.pi / 180), R_y(-az * math.pi / 180))
    return np.matmul(Rmat, inv_corr_mat)


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1,
    competely random rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`,
    they will be auto-generated.
    """
    # from
    # http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3, ))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    reflM = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return M.dot(reflM.T)


def rand_euler_rotation_matrix(nmax=10):
    euler = (np.random.uniform(size=(3, )) - 0.5) * nmax * 2 * math.pi / 360.0
    Rmat = euler_to_rot(euler)
    return Rmat, euler * 180 / math.pi


def rot_mag(R):
    angle = (1.0 / math.sqrt(2)) * \
        norm(logm(R), 'fro') * 180 / (math.pi)
    return angle


def add_noise(cams, nmax=10):
    noises = []
    rot_noise = []
    for bx in range(cams.shape[0]):
        item_max_noise = float(nmax)
        for ix in range(cams.shape[1]):
            rand_rot, euler = rand_euler_rotation_matrix(item_max_noise)
            noises.append(euler)
            rot_noise.append(rand_rot)
            R_noisy = np.matmul(cams[bx, ix, :, :3], rand_rot)
            cams[bx, ix, :, :3] = R_noisy
    return cams, rot_noise, noises
