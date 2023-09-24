import pickle
from tqdm.auto import tqdm
import os
import sys
sys.path.append(os.getcwd())
from common.preprocessing import make_sentences, make_dict
from common.huffman import HuffmanTree, code_to_id
# from common.functions import unigramsampler


data_path = 'data/1-billion-word/preprocessed/'

# # make dictionary
# word_to_id, id_to_word, vocab = make_dict(0, 99, min_count=5)
# with open(data_path+'dictionary_all.pkl', 'wb') as fw:
#     pickle.dump((word_to_id, id_to_word, vocab), fw)
# print('length of vocab: ', len(vocab))

dictionary_pkl = data_path+'dictionary_all.pkl'
with open(dictionary_pkl, 'rb') as fr:
    word_to_id, id_to_word, vocab = pickle.load(fr)
    fr.close()
print(len(word_to_id), len(id_to_word), len(vocab))

# make sentence
if not os.path.exists(data_path+'corpus/'):
    os.mkdir(data_path+'corpus/')
corpus_tot_num = 0
for file_num in tqdm(range(100)):
    corpus_list = make_sentences(word_to_id, file_num)
    corpus_tot_num += len(corpus_list)
    with open(data_path+f'corpus/corpus{file_num}.pkl', 'wb') as fw:
        pickle.dump(corpus_list, fw)
sentence_num = 0
for i in range(100):
    with open(data_path+'corpus/corpus{}.pkl'.format(i), 'rb') as fr:
        sentence_list = pickle.load(fr)
        sentence_num += len(sentence_list)

print('len: ', sentence_num)
with open(data_path+'corpus_num.pkl', 'wb') as fw:
        pickle.dump(sentence_num, fw)


words_num = 0
for word, counts in vocab.items():
    if word == '<\s>':
        continue
    else:
        words_num += counts

with open(data_path+'words_num.pkl', 'wb') as fw:
        pickle.dump(words_num, fw)
print(f'The number of total words: ', words_num)

# for HS
HS = HuffmanTree()
codes, root = HS.build(vocab)
codeIndex, code_sign = code_to_id(codes, root, vocab)
with open(data_path+'HuffmanTree.pkl', 'wb') as fw:
    pickle.dump((codeIndex, code_sign), fw)
    fw.close()
