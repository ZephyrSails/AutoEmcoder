from __future__ import division, print_function, absolute_import

from AutoencoderEmbMerger import AutoencoderEmbMerger
import argparse
import logging
import csv
import sys
import time
import tensorflow as tf

__DIMENTION__ = '__dim__'

parser = argparse.ArgumentParser(description='args for meta')
parser.add_argument('--embs', nargs='+',
                    help='dir to original embeddings')
parser.add_argument('--result', type=str)
parser.add_argument('--meta_dim', type=int, default=100,
                    help='the dimension of output meta embedding')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='autoencoder learning rate')
parser.add_argument('--num_steps', type=int, default=5000,
                    help='autoencoder learning steps')
parser.add_argument('--display_step', type=int, default=1000,
                    help='autoencoder logging steps')
parser.add_argument('--embs_type', type=str, default='vecshare')
# parser.add_argument('--i', type=int, default=0)
args = parser.parse_args()


def get_embs(embs_list):
    embs, wordsets, shapes, concat_dim = [], [], [], 0
    for file_name in embs_list:
        embs.append({})
        wordsets.append(set())
        with open(file_name, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            embs[-1][__DIMENTION__] = len(next(reader, None)) - 1
            concat_dim += embs[-1][__DIMENTION__]
            # shapes.append(len(next(reader, None)) - 1)
            len(next(reader, None))
            count = 0
            for row in reader:
                count += 1
                embs[-1][row[0]] = row[1:]
                wordsets[-1].add(row[0])
        logging.info('%s has been load: %d lines, %s rank.'
                     % (file_name, count, embs[-1][__DIMENTION__]))
    return embs, wordsets, shapes, concat_dim


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
                concat += original_embs[i][word]
            else:
                concat += [0.0 for _ in xrange(original_embs[i][__DIMENTION__])]
        concated_embs.append(concat)
        # concated_embs.append(reduce(lambda a, b: a + b, ([word] for i in )))
    return concated_embs


# python meta.py --embs '/Users/xiu/github/AutoEncoderTrail/agriculture_40.csv' '/Users/xiu/github/AutoEncoderTrail/books_40.csv' --result 'meta_5000_oct_9.csv' --display_step 1 --num_steps 5000
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    if len(args.embs) < 2:
        logging.error('--embs must include 2 or more original embeddings')

    original_embs, wordsets, _, concat_dim = get_embs(args.embs)
    wordlist = list(reduce(lambda a, b: a & b, wordsets))
    fullwordlist = list(reduce(lambda a, b: a | b, wordsets))
    concated_embs = concat_embs(original_embs, wordlist)
    full_concated_embs = full_concat_embs(original_embs, fullwordlist)

    merger = AutoencoderEmbMerger(args)
    meta, predict = merger.encode(concated_embs)
    logging.info('outputting meta embedding')
    with open(args.result, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['text'] + ['d%d' % i for i in xrange(args.meta_dim)])
        for i, word in enumerate(wordlist):
            writer.writerow([word] + list(meta[i]))
    logging.info('outputting predicted result')
    with open('predict_' + args.result, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['text'] + ['d%d' % i for i in xrange(concat_dim)])
        for i, word in enumerate(wordlist):
            writer.writerow([word] + list(predict[i]))
    logging.info('outputting concated embedding')
    with open('concat_' + args.result, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['text'] + ['d%d' % i for i in xrange(concat_dim)])
        for i, word in enumerate(wordlist):
            writer.writerow([word] + list(concated_embs[i]))
    logging.info('outputting concated all embedding')
    with open('concat_all_' + args.result, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['text'] + ['d%d' % i for i in xrange(concat_dim)])
        for i, word in enumerate(fullwordlist):
            writer.writerow([word] + list(full_concated_embs[i]))
