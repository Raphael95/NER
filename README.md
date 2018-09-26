# Named Entity Recognition base on Chinese with Bi-directional LSTM and CRF

## 1. required environment
environment |  version 
----------- | ---------
numpy       |  1.13.3
tensorflow  |  1.6.0



## 2.  description

Recurrent neural network takes as input a sequence of vectors (x1, x2, x3,,,,xn) that represents some information about sequence at every step in the input, but it fail to learn long dependencies, LSTM have designed to combat this issue by incorporating a memory-cell and have been shown to capture long-range dependencies.

Conditional Random Fileds can combine the context, for instance, the first word tag is B-PER, the second word can not be B tag, if the model do not take the crf layer into it, it may caculate the high probability that B can occur fater the B, so in order to reduce this
error,  we add a crf layer on the top of bi-lstm layer.

![text cnn](https://huangzhanpeng.github.io/2018/01/02/Neural-Architectures-for-Named-Entity-Recognition/QQ20170925-090607@2x.png)


## Reference
1. [Bidirectional LSTM-CRF Models for Sequence Tagging.  2015](https://arxiv.org/pdf/1508.01991.pdf)
2. [Neural Architectures for Named Entity Recognition.  2016](https://arxiv.org/pdf/1603.01360.pdf)

