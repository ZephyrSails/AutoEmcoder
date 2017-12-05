import sys
import numpy as np
import csv
import logging


__DIMENTION__ = '__dim__'


def get_embs(embs_list):
    embs, wordsets, shapes, concat_dim = [], [], [], 0
    for file_name in embs_list:
        embs.append({})
        wordsets.append(set())
        # print file_name
        with open(file_name, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            embs[-1][__DIMENTION__] = len(next(reader, None)) - 1
            concat_dim += embs[-1][__DIMENTION__]
            # shapes.append(len(next(reader, None)) - 1)
            len(next(reader, None))
            count = 0
            for row in reader:
                if not row:
                    continue
                count += 1
                embs[-1][row[0]] = map(float, row[1:])
                wordsets[-1].add(row[0])
        logging.info('%s has been load: %d lines, %s rank.'
                     % (file_name, count, embs[-1][__DIMENTION__]))
    return embs, wordsets, shapes, concat_dim


if __name__ == '__main__':
    embs, _, _, _ = get_embs([sys.argv[1]])
    print '%s has been loaded' % sys.argv[1]
    # print embs[0].values()
    arr = np.array(embs[0].values()[:13000])
    print arr.mean(), arr.max(), arr.min()
