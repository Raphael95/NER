import tensorflow as tf
import numpy as np
import data_process
from tensorflow.contrib import crf
import time

start_time = None
end_time = None


def sent_input():
    print("please input the sentence :")
    sent = input()
    global start_time
    start_time = time.time()
    print("sent is : ", str(sent))
    if sent == '' or sent.isspace():
        print("have a nice day !")
    else:
        sentence = list(sent.strip())
        print(sentence)
        data = [(sentence, ['O'] * len(sentence))]
        print(data)
        batch = data_process.batch_data(data, 50, max_sequence=4812)
        #batch_x, batch_y, sequence_length = batch.__next__()
        return str(sent.strip()), batch

def restore_model():
    sess = tf.Session()
    ckpt_file = tf.train.latest_checkpoint('./model/')
    saver = tf.train.import_meta_graph(ckpt_file + '.meta')
    saver.restore(sess, ckpt_file)

    return sess

def predict():
    sentence, batch = sent_input()
    sess = restore_model()
    #x = sess.graph.get_tensor_by_name("x:0")
    x = sess.graph.get_operation_by_name("x").outputs[0]
    y = sess.graph.get_tensor_by_name("y:0")
    #keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
    keep_prob = sess.graph.get_operation_by_name("keep_prob").outputs[0]
    #sequence_length = sess.graph.get_tensor_by_name("sequence_len:0")
    sequence_length = sess.graph.get_operation_by_name("sequence_len").outputs[0]

    #y_hat = sess.graph.get_tensor_by_name("output_layer/matmul/y_hat:0")
    y_hat = sess.graph.get_operation_by_name("output_layer/matmul/y_hat").outputs[0]
    #transition_param = sess.graph.get_tensor_by_name("transitions:0")
    transition_param = sess.graph.get_operation_by_name("transitions").outputs[0]
    batch_x, batch_y, sequence_len = batch.__next__()
    logits, transition_param = sess.run([y_hat, transition_param], feed_dict={x: batch_x, keep_prob: 0.5, sequence_length: sequence_len})

    viterbi_sequences = []
    for logit, sequence in zip(logits, sequence_len):
        logit = logit[: sequence]
        viterbi_sequence, _ = crf.viterbi_decode(logit, transition_param)
    viterbi_sequences.append(viterbi_sequence)

    return viterbi_sequences, sentence




if __name__ == '__main__':
    results, sentence = predict()
    print("predicted results is : ", results)
    for seq in results:
        chunks = data_process.get_chunks(seq)
        ner = {}
        print(chunks)
        for entity, s, e in chunks:
            if entity not in ner:
                ner[entity] = [sentence[s: e]]

            else:
                ner[entity].append(sentence[s: e])
        print(ner)
    end_time = time.time()
    print("耗时： %f s" % (end_time - start_time))z
    print("耗时 %f s"%)
