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
        self.activation = args.activation

    # Building single layer encoder
    def _encoder(self, x, input_rank):
        weight = tf.Variable(tf.random_normal([input_rank, self.meta_dim]))
        bias = tf.Variable(tf.random_normal([self.meta_dim]))
        # Encoder Hidden layer with sigmoid activation #1
        layer_1_before_act = tf.add(tf.matmul(x, weight), bias)
        if self.activation == None:
            layer_1 = layer_1_before_act
        if self.activation == 'sigmoid':
            layer_1 = tf.nn.sigmoid(layer_1_before_act)
        if self.activation == 'tanh':
            layer_1 = tf.nn.tanh(layer_1_before_act)

        # layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weight), bias))
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

    def _validater(self, X_valide, x_dim):
        # X_valide = tf.placeholder('float', [None, x_dim])
        encoder_op = self._encoder(X_valide, x_dim)
        decoder_op = self._decoder(encoder_op, x_dim)
        y_pred = decoder_op
        y_true = X_valide
        validate_op = tf.pow(y_true - y_pred, 2)
        return validate_op

    def encode(self, embs, testingSetSize=0):
        # Construct model
        logging.debug("Construct model")
        x = embs[testingSetSize:]
        testingSet = embs[:testingSetSize]
        logging.debug("len(x): %d; len(testingSet): %d" % (len(x), len(testingSet)))
        X = tf.placeholder('float', [None, len(x[0])])
        # X_valide = tf.placeholder('float', [None, len(x[0])])

        encoder_op = self._encoder(X, len(x[0]))
        decoder_op = self._decoder(encoder_op, len(x[0]))

        # validate_op = self._validater(X_valide, len(x[0]))
        # Prediction
        logging.debug("setup Prediction")
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = X
        # validate_op = tf.metrics.mean_tensor(tf.pow(y_true - y_pred, 2))

        # Define loss and optimizer, minimize the squared error
        logging.debug("Define loss and optimizer, minimize the squared error")
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

        # normalize_true = tf.nn.l2_normalize(y_true, 1)
        # normalize_pred = tf.nn.l2_normalize(y_pred, 1)
        # loss = tf.losses.cosine_distance(normalize_true, normalize_pred, 1)
        optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate
        ).minimize(loss)

        logging.debug("Initialize the variables")
        init = tf.global_variables_initializer()
        # Start Training
        # Start a new TF session
        l_history = []
        v_history = []

        with tf.Session() as sess:
            # Run the initializer
            logging.debug("Run the initializer")
            sess.run(init)
            # Training
            logging.debug("Training start")
            for i in xrange(self.num_steps):
                # Run backprop and cost op (to get loss value)
                # logging.debug('X: %s, feed: %s' % (X.shape, feed.shape))
                _, l = sess.run(
                    [optimizer, loss],
                    feed_dict={X: x}
                )

                # Display logs per step
                if i % self.display_step == 0:
                    logging.info('Step %i: Minibatch Loss: %f' % (i, l))
                    l_history.append(l)
                    if testingSet:
                        v = sess.run(loss, feed_dict={X: testingSet})
                        logging.info('Step %i: Validation Loss: %s' % (i, str(v)))
                        v_history.append(v)
            # rebuild middle layer with learned encoder
            midLayers = sess.run(encoder_op, feed_dict={X: embs})
            predict = sess.run(decoder_op, feed_dict={X: embs})
            # print midLayers.shape()
        return midLayers, predict, l_history, v_history
