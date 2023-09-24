import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from common.functions import sigmoid


class ModelCBOW_HS:
    def __init__(self, vocab_size, embedding_size, code_index, code_sign):
        # w_in : embedding matrix
        self.embedding_size = embedding_size
        self.code_index = code_index  # id in hierarchical softmax
        self.code_sign = code_sign  # code_sign in hierarchical softmax -> -1 또는 1
        self.W_in = 0.01 * np.random.randn(vocab_size, embedding_size).astype('f')
        self.W_out = np.zeros((vocab_size-1, embedding_size)).astype('f')

    def forward(self, center, contexts):
        h = self.W_in[contexts]
        h = np.sum(h, axis=0)
        h /= len(contexts)
        h = h.reshape(1, self.embedding_size)

        code_idx = self.code_index[center]
        sign = self.code_sign[center]
        print(h.shape)
        print(self.W_out[code_idx].shape)
        score = np.dot(h, self.W_out[code_idx].T)
        # print(score)
        score *= sign
        # print(score)
        # score는 (h.W_out[code_idx_1])*(code_sign_1), (h.W_out[code_idx_2])*(code_sign_2),... 로 path개만큼 있음

        loss = -np.sum(np.log(sigmoid(score) + 1e-07))
        # 각 path에 대해서 sigmoid 한 다음에 log취해서 더함.

        self.cache = (h, contexts, code_idx, sign, score)
        return loss, np.sum(score)

    def backward(self, lr):
        h, contexts, code_idx, sign, score = self.cache
        # backward
        dout = sigmoid(score)
        dout -= 1
        dout *= sign
        print(h.T.shape)
        print(dout.shape)
        dh = np.dot(dout, self.W_out[code_idx])       # dh = (1, embedding_size)
        dW_out = np.dot(h.T, dout).T                    # dW_out = (code_index, embedding_size)

        # update
        self.W_out[code_idx] -= dW_out * lr
        self.W_in[contexts] -= dh.squeeze()/len(contexts) * lr
        return None


