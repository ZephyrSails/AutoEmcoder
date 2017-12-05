from __future__ import division, print_function, absolute_import

import logging
import numpy as np


class SVDEmbMerger:
    def __init__(self, args):
        # the rank of hidden layer decides the rank of meta embedding
        self.meta_dim = args.meta_dim


    def encode(self, embs):
        embs = np.array(embs)

        logging.debug("SVD begin %s" % str(np.shape(embs)))

        U, _, _ = np.linalg.svd(embs, full_matrices=True)

        logging.debug("SVD done")

        return U[:, :self.meta_dim]
