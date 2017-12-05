from __future__ import division, print_function, absolute_import

import tensorflow as tf
import logging
import numpy as np


class SVDEmbMerger:
    def __init__(self, args):
        # the rank of hidden layer decides the rank of meta embedding
        self.meta_dim = args.meta_dim


    def encode(self, embs):

        U, s, V = np.linalg.svd(embs, full_matrices=True)

        return U[:, :self.meta_dim]
