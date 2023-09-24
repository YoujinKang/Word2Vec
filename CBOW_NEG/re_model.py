import argparse
import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())
from common.functions import sigmoid


parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, default='./CBOW_NEG/train_log/CBOW_Neg5_sub1e-05_window8_dim300_1epoch/weight')
parser.add_argument('--weight', type=str, default='ckpt_weight.pkl')
args = parser.parse_args()

file = os.path.join(args.log_dir, args.weight)
with open(file, 'rb') as fr:
    vectors = pickle.load(fr)
word_vectors = None


class ModelCBOW_Neg:
    def __init__(self, vocab_size, embedding_size, sample_table, negative_num):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.sample_table = sample_table
        self.negative_num = negative_num
        self.W_in = np.array(vectors[0])
        self.W_out = np.array(vectors[1])

    def forward(self, center, contexts):
        h = self.W_in[contexts]
        h = np.sum(h, axis=0)
        if len(contexts) == 0:
            pass
        else: 
            h /= len(contexts)
        h = h.reshape(1, self.embedding_size) 

        neg_samples = []
        while True:
            b = np.random.randint(low=0, high=len(self.sample_table), size=self.negative_num)
            # 0 이상의 정수로 이루어진 랜덤한 numpy array 생성. 높이는 unigram table이고 크기는 negative number
            if center in self.sample_table[b]:  # 랜덤하게 생성한 b에 대한 unigramtable에 center 단어가 있으면
                continue  # 추출안함
            else:
                neg_samples = self.sample_table[b]
                break  # else가 걸릴때까지 계속 b를 바꿔가며 찾다가 찾으면 멈춤
        label = np.append([center], neg_samples)

        # forward
        out = sigmoid(np.dot(h, self.W_out[label].T))  # 각 결과들의 레이블마다 시그모이드 적용
        p_loss = -np.log(out[:,:1] + 1e-07)  # out = (1, label) -> 0번째 레이블 = 정답 -> 1이 아니면 모두 로스
        n_loss = -np.sum(np.log(1 - out[:,1:] + 1e-07))  # 0번째 이후는 모두 오답 -> 0이 아니면 모두 로스

        self.cache = (h, contexts, label, out)
        loss = float(p_loss + n_loss)
        return loss, np.sum(out)/len(out)
        
    def backward(self, lr):
        h, contexts, label, out = self.cache

        dout = out.copy()
        dout[:,:1] -= 1  # 정답에 대한 out값  -> y-t = p_out-1
        dw_out = np.dot(h.T, dout).T
        dh = np.dot(dout, self.W_out[label])  # dout과 W_out에서 룩업해서 dot product

        # weight update
        self.W_out[label] -= dw_out * lr 
        self.W_in[contexts] -= dh.squeeze() / len(contexts) * lr
        return None
