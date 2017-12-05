from __future__ import division, print_function, absolute_import

import utils
from AutoencoderEmbMerger import AutoencoderEmbMerger
import argparse
import logging
import csv
import sys
import time
import tensorflow as tf
from sim_benchmark import _eval_all
import numpy as np


parser = argparse.ArgumentParser(description='args for meta')
parser.add_argument('--embs', nargs='+',
                    help='dir to original embeddings')
parser.add_argument('--result', type=str)
parser.add_argument('--meta_dim', type=int, default=100,
                    help='the dimension of output meta embedding')
parser.add_argument('--testing_size', type=int, default=0,
                    help='testing set size')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='autoencoder learning rate')
parser.add_argument('--num_steps', type=int, default=5000,
                    help='autoencoder learning steps')
parser.add_argument('--display_step', type=int, default=1000,
                    help='autoencoder logging steps')
parser.add_argument('--method', type=str, default='autoemcoder',
                    help='autoemcoder, svd')
parser.add_argument('--embs_type', type=str, default='vecshare')
parser.add_argument('--activation', type=str, default=None)
parser.add_argument('--meta_target', type=str, default='concat')
# parser.add_argument('--i', type=int, default=0)
args = parser.parse_args()


def concat_embs(original_embs, wordlist):
    concated_embs = []
    for word in wordlist:
        concated_embs.append(reduce(lambda a, b: a + b, (original_embs[i][word] for i in xrange(len(original_embs)))))
    return concated_embs


def full_concat_embs(original_embs, wordlist):
    concated_embs = []
    for word in wordlist:
        concat = []
        for i in xrange(len(original_embs)):
            if word in original_embs[i]:
                concat += map(float, original_embs[i][word])
            else:
                concat += [0.0 for _ in xrange(original_embs[i][__DIMENTION__])]
        concated_embs.append(concat)
    return concated_embs


def full_overlay_embs(original_embs, wordlist):
    overlay_embs = []

    for word in wordlist:
        # concat = np.array([0.0 for _ in xrange(len(original_embs[0][__DIMENTION__]))])
        concat = np.zeros(original_embs[0][__DIMENTION__])
        for i in xrange(len(original_embs)):
            if word in original_embs[i]:
                # print(concat, np.array(original_embs[i][word]))
                # print(concat)
                # print(original_embs[i][word])
                concat += np.array(map(float, original_embs[i][word]))
            # else:
                # concat += np.array([0.0 for _ in xrange(original_embs[i][__DIMENTION__])])
        overlay_embs.append(list(concat))
    return overlay_embs


def output(wordlist, embedding, file_prefix, args):
    file_name = file_prefix + args.result
    logging.info('outputting to %s' % file_name)
    with open(file_name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['text'] + ['d%d' % i for i in xrange(len(embedding[0]))])
        logging.info('emblen: %d, wlLen: %d' % (len(embedding), len(wordlist)))
        for i, word in enumerate(wordlist):
            writer.writerow([word] + list(embedding[i]))

    logging.info(_eval_all(file_name))


# python meta.py --embs '/Users/xiu/github/AutoEncoderTrail/agriculture_40.csv' '/Users/xiu/github/AutoEncoderTrail/books_40.csv' --result 'meta_5000_oct_9.csv' --display_step 1 --num_steps 5000
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    if len(args.embs) < 2:
        logging.error('--embs must include 2 or more original embeddings')

    original_embs, wordsets, _, _ = get_embs(args.embs)
    # wordlist = list(reduce(lambda a, b: a & b, wordsets))
    fullwordlist = list(reduce(lambda a, b: a | b, wordsets))
    # concated_embs = concat_embs(original_embs, wordlist)
    if args.meta_target == 'concat':
        target_embs = full_concat_embs(original_embs, fullwordlist)
    elif args.meta_target == 'overlay':
        target_embs = full_overlay_embs(original_embs, fullwordlist)

    if args.method == 'autoemcoder':
        merger = AutoencoderEmbMerger(args)
        meta, predict, l_history, v_history = merger.encode(target_embs, args.testing_size)

        output(fullwordlist, predict, 'predict_', args)
        output(fullwordlist, target_embs, 'target_' + args.meta_target + '_', args)

        with open('l_' + args.result, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for l in l_history:
                writer.writerow([l])
        with open('v_' + args.result, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for v in v_history:
                writer.writerow([v])
    elif args.method == 'svd':
        merger = AutoencoderEmbMerger(args)
        meta = merger.encode(target_embs)

    output(fullwordlist, meta, '', args)

    logging.info('All Done, Cheers!')
