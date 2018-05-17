import tensorflow as tf
import os,sys
from datetime import datetime
import numpy as np
from sklearn.metrics import roc_auc_score as auc
from sklearn.base import BaseEstimator, TransformerMixin

class AutoEncoder:
    def __init__(self,
                feature_size,
                encoder_layers,
                learning_rate,
                epochs,
                batch_size,
                random_seed,
                display_step= 1,
                verbose= True,
                model_path= '.'):
        self.feature_size = feature_size
        self.encoder_layers = encoder_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.display_step = display_step
        self.verbose = verbose
        self.model_file = os.path.join(model_path, 'auto_encoder.ckpt')

        self._init_graph()

    def _init_graph(self):
        ''''''
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.X = tf.placeholder('float32', [None, self.feature_size])
            self.weights = self._init_weights()

            # encoder
            #self.out = self.X
            for i in range(0, len(self.encoder_layers)):
                if(i == 0):
                    self.out = tf.nn.tanh(tf.add(tf.matmul(self.X, self.weights['encoder_h%s' % i]), self.weights['encoder_b%s' % i]))
                else:
                    self.out = tf.nn.tanh(tf.add(tf.matmul(self.out, self.weights['encoder_h%s' % i]), self.weights['encoder_b%s' % i]))
            # decoder
            for i in range(0, len(self.encoder_layers) - 1):
                self.out = tf.nn.tanh(tf.add(tf.matmul(self.out, self.weights['decoder_h%s' % i]), self.weights['decoder_b%s' % i]))
            self.out = tf.nn.tanh(tf.add(tf.matmul(self.out, self.weights['decoder_h%s' % (len(self.encoder_layers) - 1)]),self.weights['decoder_b%s' % (len(self.encoder_layers) - 1)]))
            # Define batch mse
            self.batch_mse = tf.reduce_mean(tf.pow(self.X - self.out, 2), 1)
            # Define loss and optimizer, minimize the squared error
            self.cost = tf.reduce_mean(tf.pow(self.X - self.out, 2))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

            # init
            self.saver = tf.train.Saver()
            var_init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(var_init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _init_weights(self):
        ''''''
        weights = {}
        num_encoder_layer = len(self.encoder_layers)
        neuro_num = [self.feature_size]
        neuro_num.extend(self.encoder_layers)
        # encoder weights
        for i in range(num_encoder_layer):
            weights['encoder_h%s' % i] = tf.Variable(tf.random_normal([neuro_num[i], neuro_num[i + 1]]))
            weights['encoder_b%s' % i] = tf.Variable(tf.random_normal([neuro_num[i + 1]]))
        # decoder weights
        for i in range(num_encoder_layer):
            weights['decoder_h%s' % i] = tf.Variable(tf.random_normal([neuro_num[-i - 1], neuro_num[-i - 2]]))
            weights['decoder_b%s' % i] = tf.Variable(tf.random_normal([neuro_num[-i - 2]]))
        #weights['decoder_h%s' % (num_encoder_layer - 1)] = tf.Variable(tf.random_normal([self.encoder_layers[0], self.feature_size]))
        #weights['decoder_b%s' % (num_encoder_layer - 1)] = tf.Variable(tf.random_normal([self.feature_size]))

        return weights

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def fit(self, X_train, y_train, X_valid, y_valid):
        ''''''
        now = datetime.now()
        total_batch = int(X_train.shape[0] / self.batch_size)
        #Training cycle
        for epoch in range(self.epochs):
            # Loop over all batches
            for j in range(total_batch):
                batch_idx = np.random.choice(X_train.shape[0], self.batch_size)
                batch_xs = X_train[batch_idx]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: batch_xs})

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                train_batch_mse = self.sess.run(self.batch_mse, feed_dict={self.X: X_train})
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c),"Train auc=", "{:.6f}".format(auc(y_train, train_batch_mse)),
                      "Time elapsed=", "{}".format(datetime.now() - now))
            print("Optimization Finished!")
        save_path = self.saver.save(self.sess, self.model_file)
        print("Model saved in file: %s" % save_path)
