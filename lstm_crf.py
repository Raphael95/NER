import tensorflow as tf
import numpy as np
import os
import data_process
from tensorflow.contrib import crf
import time

with open(os.path.abspath(os.path.dirname(os.getcwd())) + '/data/vocab_size.txt', 'r') as f:
    vocab_size = int(f.read())   # 文件读取出来的是 str, 需要转换成int, 以便rnn类初始化时读取该数据

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

log_dir = '/Users/raphael/tensor_log'

class Recurrent_Neuron(object):

    def __init__(self):
        self.sequence_dim = 4812
        self.hidden_dim = 80
        self.output_dim = 15

        self.embedding_dim = 200
        self.vocab_size = vocab_size

        self.learning_rate = 0.01
        self.batch_size = 50
        self.iter_size = 800

    def add_placeholder(self):

        self.x = tf.placeholder(tf.int32, shape=[None, self.sequence_dim], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None, self.sequence_dim], name='y')

        self.droput = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.sequence_len = tf.placeholder(tf.int32, shape=[None], name='sequence_len')


    def embedding_layer(self):

        with tf.name_scope("embedding_layer"):
            embedding = tf.Variable(tf.truncated_normal(shape=[self.vocab_size, self.embedding_dim], stddev=0.1))
            embed = tf.nn.embedding_lookup(params=embedding, ids=self.x)
            self.embedded = tf.nn.dropout(tf.reshape(embed, [-1, self.sequence_dim, self.embedding_dim]),
                                          keep_prob=self.droput)


    def lstm_layer(self):

        with tf.name_scope("lstm_layer"):
            fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_dim, forget_bias=1.0, state_is_tuple=True)
            bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_dim, forget_bias=1.0, state_is_tuple=True)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw, cell_bw=bw, inputs=self.embedded,
                                                                        sequence_length=self.sequence_len,
                                                                        dtype=tf.float32)
            outputs = tf.concat([output_fw, output_bw], axis=-1)
            self.output = tf.nn.dropout(outputs, keep_prob=self.droput)

    def output_layer(self):

        with tf.name_scope("output_layer"):

            with tf.name_scope("weight"):
                weight = tf.Variable(tf.truncated_normal(shape=[2 * self.hidden_dim, self.output_dim], stddev=0.1))

            with tf.name_scope("bias"):
                bias = tf.Variable(tf.constant(0.1, shape=[self.output_dim]))

            with tf.name_scope("matmul"):
                output = tf.reshape(self.output, [-1, 2 * self.hidden_dim])
                outputs = tf.matmul(output, weight) + bias
                self.y_hat = tf.reshape(outputs, [-1, self.sequence_dim, self.output_dim], name='y_hat')
                print("y_hat is :", self.y_hat.name)

    def loss_function(self):

        with tf.name_scope("loss"):
            log_likelihood, self.transition_params = crf.crf_log_likelihood(inputs=self.y_hat, tag_indices=self.y,
                                                                            sequence_lengths=self.sequence_len)
            self.loss = -tf.reduce_mean(log_likelihood)
            print("transition_params name is : ", self.transition_params.name)  # transitions:0

        tf.summary.scalar("log_loss", self.loss)


    def init_op(self):
        tf.global_variables_initializer().run()


    def add_summary(self):

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(log_dir + '/train/lstm', sess.graph)

    def add_optimizer(self):

        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def save_model(self):
        self.saver = tf.train.Saver(max_to_keep=1)

    def predict_batch(self, batch_x, sequence_len):
        """

        :param batch_x: batch_x
        :param sequence_len:  sequence len : [10, 20,40...24]
        :return: pred_label : [0, 1, 5, 0, 0, 0, 4, 2, 8, 12], sequence_len
        """


        viterbi_sequences = []
        logits, transition_params = sess.run([self.y_hat, self.transition_params], feed_dict={self.x: batch_x,
                                                                                              self.droput: 0.5,
                                                                                              self.sequence_len: sequence_len})
        # iterator over the sentence
        for logit, sequence_length in zip(logits, sequence_len):

            # keep only the valid time step
            logit = logit[: sequence_length]
            viterbi_sequence, viterbi_score = crf.viterbi_decode(logit, transition_params)
            print("viterbi_sequence is :", viterbi_sequence)
            print("sequence len is :", sequence_length)
            viterbi_sequences.append(viterbi_sequence)
        print("sequence is :", viterbi_sequences)
        print("batch length is :", len(viterbi_sequences))
        return viterbi_sequences, sequence_len


    def evaluate(self, batch_x, batch_y, sequence_len):
        """
        :param batch_x:  batch_x
        :param batch_y:  batch_y
        :param sequence_len:  sequecne len : [10, 38, 24, 10..., 35]
        :return:
        """

        accuracys = []

        correct_predicts, total_correct, total_preds = 0., 0., 0.
        label_pred, sequence_length = self.predict_batch(batch_x, sequence_len)
        for label, label_pred, length in zip(batch_y, label_pred, sequence_length):
            label = label[: length]
            label_pred = label_pred[: length]

            accuracys += map(lambda x: x[0] == x[1], zip(label, label_pred))
            # print("accuracys is :", accuracys)  #计算每个轮回的准确率

        accu = np.mean(accuracys)
        print("accu is :", accu)
        tf.summary.scalar("accuracy", accu)
        return accu



    def feed_data(self):

        accuracy = 0.0
        data = data_process.read_corpus()
        batch = data_process.batch_data(data, self.batch_size)
        for i in range(self.iter_size):
            batch_x, batch_y, sequence_len = batch.__next__()
            if i % 100 == 0:

                # aa = sess.run(self.embedded, feed_dict={self.x: batch_x, self.y: batch_y, self.droput: 0.5,
                #                                         self.sequence_len: sequence_len})
                # bb = tf.reshape(aa, [-1, self.sequence_dim, self.embedding_dim])
                # print(bb.shape)
                # print(sess.run(bb))

                summary, y_hat, loss = sess.run([self.merged, self.y_hat, self.loss], feed_dict={self.x: batch_x,
                                                                                                 self.y: batch_y,
                                                                                                 self.droput: 0.5,
                                                                                                 self.sequence_len: sequence_len})
                accu = self.evaluate(batch_x, batch_y, sequence_len)
                self.train_writer.add_summary(summary, i)
                print("y_hat is :", y_hat.shape)
                print("log loss is : ", loss)
                if accu > accuracy:
                    self.saver.save(sess, './model/lstm', global_step=i)
                    accuracy = accu

            self.train_step.run(feed_dict={self.x: batch_x, self.y: batch_y, self.droput: 0.5,
                                           self.sequence_len: sequence_len})


    def build_graph(self):
        self.add_placeholder()
        self.embedding_layer()
        self.lstm_layer()
        self.output_layer()
        self.loss_function()
        self.add_optimizer()
        self.add_summary()
        self.init_op()
        self.save_model()
        self.feed_data()


if __name__ == '__main__':

    begin = time.time()
    rnn = Recurrent_Neuron()
    rnn.build_graph()
    print("耗时 ：%f m " % ((time.time() - begin) / 60))

