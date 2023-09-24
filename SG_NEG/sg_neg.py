import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from common.functions import sigmoid, make_contexts, unigramsampler


class ModelSG_Neg:
    def __init__(self, vocab_size, embedding_size, sample_table, negative_num):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.sample_table = sample_table
        self.negative_num = negative_num
        self.W_in = 0.01 * np.random.randn(vocab_size, embedding_size).astype('f')
        self.W_out = np.zeros((vocab_size, embedding_size)).astype('f')

    def forward(self, center, contexts):
        h = self.W_in[center]

        tot_label = []
        neg_samples = []
        for context in contexts:
            while True:
                b = np.random.randint(low=0, high=len(self.sample_table), size=self.negative_num)
                # 0 이상의 정수로 이루어진 랜덤한 numpy array 생성. 높이는 unigram table이고 크기는 negative number
                for samples in self.sample_table[b]:
                    if samples in contexts:
                        break
                else:
                    neg_samples = self.sample_table[b]
                    break
            label = np.append(context, neg_samples)
            tot_label.append(label)
        # print('tot_label: ', tot_label)
        # forward
        w_out = np.zeros((len(contexts), self.embedding_size, self.negative_num+1)).astype('f')
        for i in range(len(contexts)):
            for j in range(len(tot_label)):
                w_out[i] = self.W_out[tot_label[j]].T
        # print('h shape: ', h.shape)
        # print('w_out shape: ', w_out.shape)
        out = sigmoid(np.dot(h, w_out)) # 각 결과들의 레이블마다 시그모이드 적용
        # print('out= ', out)  # out: (len(contexts), negative_num + 1))
        loss = []
        for i in range(len(contexts)):
            p_loss = -np.log(out[i][:1] + 1e-07)  # out = (1, label) -> 0번째 레이블 = 정답 -> 1이 아니면 모두 로스
            n_loss = -np.sum(np.log(1 - out[i][1:] + 1e-07))  # 0번째 이후는 모두 오답 -> 0이 아니면 모두 로스
            loss_temp = float(p_loss + n_loss)
            loss.append(loss_temp)
        loss = np.sum(loss)

        self.cache = (h, center, tot_label, out, len(contexts), w_out)

        return loss, np.sum(out)/len(out)

    def backward(self, lr):

        h, center, tot_label, out, len_contexts, w_out = self.cache

        dout = out.copy()
        for i in range(len_contexts):
            dout[i][:1] -= 1  # 정답에 대한 out값  -> y-t = p_out-1

        dout = dout.reshape(len_contexts, 1, self.negative_num+1)
        h = h.reshape(self.embedding_size, 1)

        dh = np.zeros((len_contexts, self.embedding_size))
        for i in range(len_contexts):
            dh[i] = np.matmul(dout[i], w_out.transpose(0, 2, 1)[i])

        dW_out = np.dot(h, dout.T).T  # dW_out : (len(contexts), negative_num+1 embedding_size)
        # print(dW_out.shape)

        # print('tot_label: ', tot_label)
        for i in range(len_contexts):
            for j in range(len(tot_label)):
                self.W_out[tot_label[j]] -= dW_out[i] * lr
            self.W_in[center] -= dh[i] * lr

        return None


def test():
    word_to_id = {'<\s>': 0, 'bb': 1, 'cc': 2, 'dd': 3, 'ee': 4, 'aa': 5}
    vocab = {'<\s>': 0, 'aa': 1, 'bb': 6, 'cc': 3, 'dd': 2, 'ee': 2}
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
    negative_num = 2
    sample_table = unigramsampler(vocab, word_to_id, power=3 / 4)
    sample_table = np.array(sample_table)
    model = ModelSG_Neg(V, embedding_size, sample_table, negative_num)
    loss, score = model.forward(center, contexts)
    print('loss: ', loss)
    print('score: ', score)
    model.backward(0.1)



# test()

