import numpy as np
import argparse
import pickle
import os
import sys
sys.path.append(os.getcwd())
from common.functions import sigmoid, make_contexts
from common.huffman import HuffmanTree, code_to_id


class ModelSG_HS:   # center로부터 주변단어 예측
    def __init__(self, vocab_size, embedding_size, code_index, code_sign, code_index_len):
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_dir', type=str, default='./SG_HS/train_log/SG_HS_sub1e-05_window10_dim300_3epoch/weight')
        parser.add_argument('--weight', type=str, default='SG_HS_sub1e-05_window10_dim300_3epoch.pkl')
        args = parser.parse_args()

        # load word vectors
        file = os.path.join(args.log_dir, args.weight)
        with open(file, 'rb') as fr:
            vectors = pickle.load(fr)

        # w_in : embedding matrix
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.code_index = code_index  # id in hierarchical softmax
        self.code_sign = code_sign  # code_sign in hierarchical softmax -> -1 또는 1
        self.code_len = code_index_len  # code_index가 모두 같은 크기로 통일되어있음
        self.W_in = np.array(vectors[0])
        self.W_out = np.array(vectors[1])

    def forward(self, center, contexts):
        h = self.W_in[center]
        # h = h.reshape(1, self.embedding_size)  # h: (1, embedding_size)

        code_idx = [self.code_index[context] for context in contexts]

        # print('code_idx: ', code_idx)
        sign = [self.code_sign[context] for context in contexts]
        # print('sign: ', sign)

        # hx = h[np.newaxis, :]  # hx: (1, 1, embedding_size)
        w_out = np.zeros((len(contexts), self.embedding_size, self.code_len)).astype('f')
        
        for i in range(len(contexts)):
            w_out[i] = self.W_out[code_idx[i]].T
        #
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


# test()

