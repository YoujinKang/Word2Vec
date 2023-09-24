# from sg_hs import ModelSG_HS
from sg_hs_remodel import ModelSG_HS
import argparse
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm.auto import tqdm
import time
import json
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from common.functions import subsampling_prob, subsampling, make_contexts

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='SG_HS')
parser.add_argument('--least_freq', type=int, default=5)
parser.add_argument('--sub_t', type=float, default=1e-05)
parser.add_argument('--window_size', type=int, default=10)
parser.add_argument('--max_epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.025)  # 논문값
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--eval_interval', type=int, default=10000)

args = parser.parse_args()
log_dir = '{}/train_log/{}_sub{}_window{}_dim{}_{}epoch/'.format(args.model, args.model, args.sub_t, args.window_size,
                                                                      args.embedding_size, args.max_epoch)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(log_dir + 'argparse.json', 'w') as f:
    json.dump(args.__dict__, f)

tb_dir = os.path.join(log_dir, 'tb')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)

tb_writer = SummaryWriter(tb_dir)

data_path = 'data/1-billion-word/preprocessed/'

# load dictionary
dictionary_pkl = data_path + 'dictionary_all.pkl'
with open(dictionary_pkl, 'rb') as fr:
    word_to_id, id_to_word, vocabulary = pickle.load(fr)
    fr.close()
print('Length of vocab:', len(vocabulary))
V = len(vocabulary)

# load data
with open(data_path + 'corpus_num.pkl', 'rb') as fr:
    sentence_num = pickle.load(fr)
    fr.close()
print('The number of corpus: ', sentence_num)

# count total training words.
with open(data_path + 'words_num.pkl', 'rb') as fr:
    total_word = pickle.load(fr)
    fr.close()
print('The number of total words: ', total_word)

# probability of subsampling
id_to_sub_p = subsampling_prob(vocabulary, word_to_id, total_word, args.sub_t)

# Huffman Tree
with open(data_path + 'HuffmanTree.pkl', 'rb') as fr:
    code_index, code_sign = pickle.load(fr)
    fr.close()
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


model = ModelSG_HS(V, args.embedding_size, code_index, code_sign, code_index_len)

# training start
num_trained_word = 0
current = 0
# print("start lr: ", (1 - current / (sentence_num * args.max_epoch))*args.lr)
total_score = 0
total_loss = 0
loss_count = 0
avg_loss = 0
avg_score = 0

time_list = []
start_t = time.time()
for epoch in range(3, args.max_epoch):
    print("epoch: %d/%d" % (epoch + 1, args.max_epoch))
    # per dataset segment
    if epoch == 1:
        range_start = 40
    else: 
        range_start = 0
    for j in range(range_start, 100):
        with open(data_path + 'corpus/corpus{}.pkl'.format(j), 'rb') as fr:
            sentence_list = pickle.load(fr)
            fr.close()
        # per sentence
        for i in tqdm(range(len(sentence_list)), desc='file {}/100'.format(j + 1), bar_format="{l_bar}{bar:10}{r_bar}"):
            current += 1
            # 각 문장마다 subsmpling 해서 학습시킬 문장 따로 뺌
            sentence = subsampling(sentence_list[i], id_to_sub_p)
            if not sentence:
                continue

            # per center word
            for center_idx, center in enumerate(sentence):

                # learning rate decay - Linear
                # alpha = 1 - current / (sentence_num * args.max_epoch)
                # if alpha <= 0.00001:
                    # alpha = 0.00001
                alpha = 0.00001
                lr = args.lr * alpha

                # sentence 안에서 고른 center에 맞게 window_size만큼 앞뒤로 contexts 만듦
                contexts = make_contexts(args.window_size, sentence, center_idx)
                if not contexts:
                    continue

                # training
                loss, score = model.forward(center, contexts)
                model.backward(lr)

                total_loss += loss
                total_score += score
                loss_count += 1

                num_trained_word += 1
                # 학습한 중심 단어의 갯수가 eval_interval 배수가 될 때마다 loss와 score평균, lr 저장
                if (args.eval_interval is not None) and (num_trained_word % args.eval_interval == 1):
                    elapsed_t = time.time() - start_t
                    avg_loss = total_loss / loss_count
                    avg_score = total_score / loss_count

                    total_loss = 0
                    total_score = 0
                    loss_count = 0

                    tb_writer.add_scalar('score/real_train_word(*{})'.format(args.eval_interval), avg_score,
                                         num_trained_word)
                    tb_writer.add_scalar('loss/real_train_word(*{})'.format(args.eval_interval), avg_loss,
                                         num_trained_word)
                    tb_writer.add_scalar('lr/real_train_word(*{})'.format(args.eval_interval), lr, num_trained_word)
                tb_writer.flush()

        # save temp weight per dataset segment
        print('avg_loss: ', avg_loss, 'avg_score: ', avg_score)

        weight_dir = os.path.join(log_dir, 'weight')
        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)
        with open(os.path.join(weight_dir, f'ckpt_weight{j+1}.pkl'
                .format(args.sub_t, epoch + 1)), 'wb') as fw:
            pickle.dump((model.W_in, model.W_out), fw)
        if os.path.exists(weight_dir+f'/ckpt_weight{j}.pkl'):
            os.remove(weight_dir+f'/ckpt_weight{j}.pkl')

        et = (time.time() - start_t) / 3600
        time_list.append(et)
        print("epoch: {}/{}, file {}, elapsed_time: {}[h]".format(epoch + 1, args.max_epoch, str(j + 1) + '/100', et))

    print("The number of trained words per epoch: ", num_trained_word)

    # save the weights
    weight_dir = os.path.join(log_dir, 'weight')
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    with open(os.path.join(weight_dir, '{}_sub{}_window{}_dim{}_{}epoch.pkl'
            .format(args.model, args.sub_t, args.window_size, args.embedding_size, epoch + 1)), 'wb') as fw:
        pickle.dump((model.W_in, model.W_out), fw)
        fw.close()