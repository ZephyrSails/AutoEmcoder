from __future__ import division, print_function, absolute_import

import logging
import numpy as np


class SVDEmbMerger:
    """
    python meta.py
    --embs '/Users/xiu/github/AutoEncoderTrail/agriculture_40.csv' '/Users/xiu/github/AutoEncoderTrail/books_40.csv'
    --result 'svd_100.csv' --display_step 1 --num_steps 7000 --testing_size 0
    --activation 'tanh' --method 'svd'
    """
    def __init__(self, args):
        # the rank of hidden layer decides the rank of meta embedding
        self.meta_dim = args.meta_dim


    def encode(self, embs):
        embs = np.array(embs)

        logging.debug("SVD begin %s" % str(np.shape(embs)))

        U, _, _ = np.linalg.svd(embs, full_matrices=True)

        logging.debug("SVD done")

        return U[:, :self.meta_dim]
