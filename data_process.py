import numpy as np
import os
import time
import pickle

tag2label = {'O': 0,
             'B-TIM': 1, 'I-TIM': 2,
             'B-LOC': 3, 'I-LOC': 4,
             'B-ORG': 5, 'I-ORG': 6,
             'B-COM': 7, 'I-COM': 8,
             'B-PRO': 9, 'I-PRO': 10,
             'B-JOB': 11, 'I-JOB': 12,
             'B-PER': 13, 'I-PER': 14}

def read_corpus():
    """
    :return:  data:[(['你', '好', '今', '天'], [O, O, B-TIM, I-TIM]),
                    (['腾', '讯', '股', '票'], [B-COM, I-COM, O, O])]
    """
    data = []
    sentence, tags = [], []
    with open(os.path.abspath(os.path.dirname(os.getcwd())) + '/data/train.txt', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if line != '\n':
            [sent, tag] = line.strip().split()
            sentence.append(sent)
            tags.append(tag)
        else:
            data.append((sentence, tags))
            sentence, tags = [], []
    return data

def build_vocab(data):
    """

    :param data: [(['你', '好', '今', '天'], [O, O, B-TIM, I-TIM]),
                  (['腾', '讯', '股', '票'], [B-COM, I-COM, O, O])]
    :return:  vocab : {'今': 0, '天': 1 ...}
    """
    vocab = {}
    if os.path.exists(os.path.abspath(os.path.dirname(os.getcwd())) + '/data/word2id.pkl'):
        with open(os.path.abspath(os.path.dirname(os.getcwd())) + '/data/word2id.pkl', 'rb') as f:
            vocab = pickle.load(f)
    else:
        index = 2
        for sentence, tags in data:
            for sent in sentence:
                if sent not in vocab:
                    vocab[sent] = index
                    index += 1
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        with open(os.path.abspath(os.path.dirname(os.getcwd())) + '/data/word2id.pkl', 'wb') as f:
            pickle.dump(vocab, f)

    with open(os.path.abspath(os.path.dirname(os.getcwd())) + '/data/vocab_size.txt', 'w') as f:
        f.write(str(len(vocab)))

    return vocab


def sentence2id(sentence, vocabulary):
    """

    :param sentence:  ['腾', '讯', '股', '票']
    :param vocabulary: {'今': 0, '天': 1 ...}
    :return: sent_id : [19, 32, 56, 21]
    """
    sent_id = []
    for char in sentence:
        if char not in vocabulary:
            char = '<UNK>'
        sent_id.append(vocabulary[char])
    return sent_id


def max_sequence_len(data, vocab):
    """

    :param data: [(['你', '好', '今', '天'], [O, O, B-TIM, I-TIM]),
                  (['腾', '讯', '股', '票'], [B-COM, I-COM, O, O])]
    :param vocab: {'今': 0, '天': 1 ...}
    :return:  max_len
    """
    sequence = []
    for sentence, _ in data:
        sentence_id = sentence2id(sentence, vocab)
        sequence.append(sentence_id)

    max_len = max(map(lambda x: len(x), sequence))
    return max_len


def pad_sent(sentence, max_length):
    """

    :param sentence:  [19, 32, 56, 21]
    :param max_length:
    :return: new_sentence: [19, 32, 56, 21, 0, 0, 0, ...., 0]
    """

    new_sentence = []
    current_len = len(sentence)
    if current_len <= max_length:
        diff = max_length - current_len
        new_sentence = sentence + [0 for i in range(diff)]
    return new_sentence


def batch_data(data, batch_size, shuffle=False, max_sequence=None):
    """

    :param data:  [(['你', '好', '今', '天'], ['O', 'O', 'B-TIM', 'I-TIM']),
                    (['腾', '讯', '股', '票'], ['B-COM', 'I-COM', 'O', 'O'])]
    :param batch_size:
    :param shuffle:
    :return:  batch_x, batch_y, sequence_len
    """

    vocab = build_vocab(data)
    batch_x, batch_y = [], []
    sequence_len = []

    if shuffle:
        data = np.random.shuffle(data)

    if max_sequence == None:
        max_sequence = max_sequence_len(data, vocab)
        print(max_sequence)

    for sentence, tags in data:
        if len(batch_x) == batch_size:
            yield np.array(batch_x), np.array(batch_y), sequence_len
            batch_x, batch_y, sequence_len = [], [], []

        sent = sentence2id(sentence, vocab)
        sequence_len.append(len(sent))
        #print("sequence_len is : ", sequence_len)
        sent = pad_sent(sent, max_sequence)
        #print("sent is : ", sent)

        new_tags = tags + ['O' for i in range(len(sent) - len(tags))]
        label = [tag2label[tag] for tag in new_tags]
        #print("label is : ", label)

        batch_x.append(sent)
        batch_y.append(label)


    if len(batch_x) != 0:
        yield np.array(batch_x), np.array(batch_y), sequence_len


def get_chunks(seq):
    """
    :param seq:  [19, 32, 56, 21]
    :param tag2label:  tag2label dictionary
    :return:  list of (chunk_type, chunk_start, chunk_end)
                      [('PRO', 6, 15)，('TIM', 18, 24)]
    """

    index_to_tag = {index: tag for tag, index in tag2label.items()}
    chunks = []
    chunk_start = 0
    name_entity = ''
    sign = ''
    is_start = False
    for i, label in enumerate(seq):

        tag = index_to_tag[label]
        a, ner = split_tag(tag)

        if not is_start:
            if a == 'B':
                chunk_start = i
                is_start = True
                continue

        if is_start:
            if name_entity == '':
                name_entity = ner
                sign = a
            if a != sign:
                chunks.append((name_entity, chunk_start, i))
                is_start = False
                name_entity = ''
                if a == 'B':
                    chunk_start = i
                    is_start = True
                    continue
    return chunks



def split_tag(tag):
    """
    :param tag:  tag of sequence : 'B-PER'
    :return:  split of tag: B and PER
    """
    if tag == 'O':
        a, b = 'O', 'O'
        return a, b
    else:
        a, b = tag.split('-')
        return a, b




if __name__ == '__main__':
    begin_time = time.time()
    data = read_corpus()
    print(data[0: 50])
    batch = batch_data(data, 10)
    for i in range(10):
        batch_x, batch_y, sequence_len = batch.__next__()
        print(batch_x)
        print(batch_y)
        print(sequence_len)

    print("耗时 : %f s" %((time.time() - begin_time)))

    # seq = [0, 0, 0, 0, 0, 0, 9, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0, 1, 2, 2, 2, 2, 2, 0, 0, 0]
    # sentence = '系列名称叫做《聆听报告谈体会》，是从10月20日开始，'
    # chunks = get_chunks(seq)
    # ner = {}
    # print(chunks)
    # for entity, s, e in chunks:
    #     if entity not in ner:
    #         ner[entity] = [sentence[s: e]]
    #
    #     else:
    #         ner[entity].append(sentence[s: e])
    # print(ner)






