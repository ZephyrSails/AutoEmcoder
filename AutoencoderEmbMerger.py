from __future__ import division, print_function, absolute_import

import tensorflow as tf
import logging
import numpy as np


class AutoencoderEmbMerger:
    def __init__(self, args):
        # the rank of hidden layer decides the rank of meta embedding
        self.meta_dim = args.meta_dim
        self.learning_rate = args.learning_rate
        self.num_steps = args.num_steps
        self.display_step = args.display_step

    # Building single layer encoder
    def _encoder(self, x, input_rank):
        weight = tf.Variable(tf.random_normal([input_rank, self.meta_dim]))
        bias = tf.Variable(tf.random_normal([self.meta_dim]))
        # Encoder Hidden layer with sigmoid activation #1
        # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weight), bias))
        layer_1 = tf.add(tf.matmul(x, weight), bias)
        return layer_1

    # Building single layer decoder
    def _decoder(self, x, input_rank):
        weight = tf.Variable(tf.random_normal([self.meta_dim, input_rank]))
        bias = tf.Variable(tf.random_normal([input_rank]))
        # Decoder Hidden layer with sigmoid activation #1
        # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weight), bias))
        layer_1 = tf.add(tf.matmul(x, weight), bias)
        return layer_1

    def _mergeEmbs(self, embs, word):
        x = []
        for emb in embs:
            if word in emb:
                x += emb[word]
            else:
                x += [0.0 for _ in xrange(emb['__rank__'])]
        return x

    def encode(self, embs):
        # Construct model
        logging.debug("Construct model")
        x = embs
        X = tf.placeholder('float', [None, len(x[0])])

        encoder_op = self._encoder(X, len(x[0]))
        decoder_op = self._decoder(encoder_op, len(x[0]))

        # Prediction
        logging.debug("setup Prediction")
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = X

        # Define loss and optimizer, minimize the squared error
        logging.debug("Define loss and optimizer, minimize the squared error")
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate
        ).minimize(loss)

        logging.debug("Initialize the variables")
        init = tf.global_variables_initializer()
        # Start Training
        # Start a new TF session
        with tf.Session() as sess:
            # Run the initializer
            logging.debug("Run the initializer")
            sess.run(init)
            # Training
            logging.debug("Training start")
            for i in xrange(self.num_steps):
                # Run backprop and cost op (to get loss value)
                # feed = np.array([x])
                # logging.debug('X: %s, feed: %s' % (X.shape, feed.shape))
                _, l = sess.run(
                    [optimizer, loss],
                    feed_dict={X: x}
                )
                # Display logs per step
                if i % self.display_step == 0:
                    logging.info('Step %i: Minibatch Loss: %f' % (i, l))

            # rebuild middle layer with learned encoder
            midLayers = sess.run(encoder_op, feed_dict={X: x})
            predict = sess.run(decoder_op, feed_dict={X: x})
            # print midLayers.shape()
        return midLayers, predict
