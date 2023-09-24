import argparse
import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())
from common.functions import sigmoid


parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, default='./CBOW_HS/train_log/CBOW_HS_sub1e-05_window8_dim300_1epoch/weight')
parser.add_argument('--weight', type=str, default='ckpt_weight.pkl')
args = parser.parse_args()

file = os.path.join(args.log_dir, args.weight)
with open(file, 'rb') as fr:
    vectors = pickle.load(fr)
word_vectors = None


class ModelCBOW_HS:
    def __init__(self, vocab_size, embedding_size, codeIndex, code_sign):
        # w_in : embedding matrix
        self.embedding_size = embedding_size
        self.codeIndex = codeIndex  # id in hierarchical softmax
        self.code_sign = code_sign  # code_sign in hierarchical softmax -> -1 또는 1
        self.W_in = np.array(vectors[0])
        self.W_out = np.array(vectors[1])

    def forward(self, center, contexts):
        h = self.W_in[contexts]
        h = np.sum(h, axis=0)
        h /= len(contexts)
        h = h.reshape(1, self.embedding_size) 

        code_index = self.codeIndex[center]
        sign = self.code_sign[center]
        score = np.dot(h, self.W_out[code_index].T)         # (1, path)
        score *= sign
        loss = -np.sum(np.log(sigmoid(score) + 1e-07))

        self.cache = (h, contexts, code_index, sign, score)
        return loss, np.sum(score)

    def backward(self, lr):
        h, contexts, code_index, sign, score = self.cache
        # backward
        dout = sigmoid(score)
        dout -= 1
        dout *= sign
        dh = np.dot(dout, self.W_out[code_index])       # dx = (1, D)
        dW_out = np.dot(h.T, dout).T                    # dW_out = (code_index, D)

        # update
        self.W_out[code_index] -= dW_out * lr
        self.W_in[contexts] -= dh.squeeze()/len(contexts) * lr
        return None


