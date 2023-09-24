import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from common.functions import sigmoid, make_contexts
from common.huffman import HuffmanTree, code_to_id


class ModelSG_HS:   # center로부터 주변단어 예측
    def __init__(self, vocab_size, embedding_size, code_index, code_sign):
        # w_in : embedding matrix
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.code_index = code_index  # id in hierarchical softmax
        self.code_len = len(code_index[0])  # code_index가 모두 같은 크기로 통일되어있음
        self.code_sign = code_sign  # code_sign in hierarchical softmax -> -1 또는 1
        self.W_in = 0.01 * np.random.randn(vocab_size, embedding_size).astype('f')
        self.W_out = np.zeros((self.vocab_size, self.embedding_size)).astype('f')
        # W_out: code_index에서 모자라는 애들 채워줄려고 cbow와 비교할때 사이즈 하나 키움

    def forward(self, center, contexts):
        h = self.W_in[center]
        # h = h.reshape(1, self.embedding_size)  # h: (1, embedding_size)

        code_idx = [self.code_index[context] for context in contexts]
        # print('code_idx: ', code_idx)
        sign = [self.code_sign[context] for context in contexts]
        # print('sign: ', sign)

        print("W_out shape: ", self.W_out[code_idx[0]].shape)

        w_out = np.zeros((len(contexts), self.embedding_size, self.code_len)).astype('f')
        print(w_out[0].shape)

        for i in range(len(contexts)):
            w_out[i] = self.W_out[code_idx[i]].T


        # print('h shape: ', h.shape)
        # print('w_out shape: ', w_out.shape)

        score = np.dot(h, w_out)  # score: (len(contexts), code_len)
        score *= sign
        # print('score shape: ', score.shape)

        loss = - np.sum(np.log(sigmoid(score)))
        # print('loss: ', loss)
        # numpy array 3차원짜리 (0,1,2) axis 에서 2와 1axis를 서로 바꾸고싶을 때 사용.
        self.cache = (h, center, code_idx, sign, score, len(contexts), w_out)

        return np.sum(loss)/len(contexts), np.sum(score)

    def backward(self, lr):
        h, center, code_idx, sign, score, len_contexts, w_out = self.cache
        # backward
        dout = sigmoid(score)
        dout -= 1
        dout *= sign  # dout : (len(contexts), code_len)  =  score shape과 같음
        # print('dout shape: ', dout.shape)
        # print('w_out.transpose shape: ', w_out.transpose(0, 2, 1).shape)

        dh = np.zeros((len_contexts, self.embedding_size))
        for i in range(len_contexts):
            dh[i] = np.matmul(dout[i], w_out.transpose(0, 2, 1)[i])
        # print('dh shape: ', dh.shape)


        # reshaping two matrices to dot
        dout = dout.reshape(len_contexts, 1, self.code_len)
        h = h.reshape(self.embedding_size, 1)
        # print(dout.shape)
        # print(h.shape)

        dW_out = np.dot(h, dout.T).T  # dW_out : (len(contexts), code_len, embedding_size)
        # print('dW_out shape: ', dW_out.shape)
        # print(self.W_out[code_idx[0]].shape)

        for i in range(len_contexts):
            self.W_out[code_idx[i]] -= dW_out[i] * lr
            self.W_in[center] -= dh[i] * lr

        return None


def test():
    word_to_id = {'<\s>': 0, 'b': 1, 'c ': 2, 'd': 3, 'e': 4, 'a': 5}
    vocab = {'<\s>': 0, "a": 1, 'b': 6, 'c': 3, 'd': 2, 'e': 2}
    huffman = HuffmanTree()
    char_to_code, root = huffman.build(vocab)
    code_index, code_sign = code_to_id(char_to_code, root, vocab)
    code_len = []
    for i in range(len(code_index)):
        code_len.append(len(code_index[i]))
    code_index_len = max(code_len)
    for i in range(len(code_index)):
        if len(code_index[i]) >= code_index_len:
            continue
        elif len(code_index[i]) < code_index_len:
            code_index[i].extend([code_index_len+1]*(code_index_len - len(code_index[i])))
            code_sign[i].extend([0]*(code_index_len - len(code_sign[i])))

    print('code_index_len: ', code_index_len)
    print("code_indice: ", code_index)
    print("code_signs: ", code_sign)
    sentence = [5, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    V = len(vocab)
    embedding_size = 10
    window_size = 2
    target_idx = 1
    center = sentence[target_idx]
    while True:
        contexts = make_contexts(window_size, sentence, target_idx)
        if len(contexts) != 0:
            break
    print("center: ", center, 'contexts: ', contexts)
    lr = 0.1
    model = ModelSG_HS(V, embedding_size, code_index, code_sign)
    loss, score = model.forward(center, contexts)
    print(loss, score)
    model.backward(0.1)
    # loss1, score2 = model.forward(center, contexts)
    # print(loss1, score2)


test()

