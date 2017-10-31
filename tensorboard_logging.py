import logging
from StringIO import StringIO

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class TensorboardLogger(object):
    """Logging in tensorboard without tensorflow ops.
       Adapted from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, writer=None, log_dir=None):
        """Creates a summary writer logging to log_dir."""
        self.logger = logging.getLogger('mview3d.' + __name__)
        if writer is not None:
            self.writer = writer
        else:
            if log_dir is not None:
                self.writer = tf.summary.FileWriter(log_dir, flush_secs=30)
            else:
                self.logger.error(
                    'At least one of writer or log_dir has to be not None')
                self.writer = None

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1])
            # Create a Summary value
            im_summaries.append(
                tf.Summary.Value(tag='%s/%d' % (tag, nr), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
