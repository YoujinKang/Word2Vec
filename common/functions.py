import numpy as np
from tqdm.auto import tqdm


def sigmoid(x):
    out = 1 / (1 + np.exp(-x))
    return out

def subsampling_prob(vocabulary, word_to_id, total_word, sub_t):
    key = []
    idx = []
    prob = []
    sub_p = []
    for (word, f) in vocabulary.items():
        key.append(word)
        idx.append(word_to_id[word])
        prob.append(f/total_word)
    for p in prob:
        sub_p.append((1+np.sqrt(p/sub_t)) * sub_t/p)
    id_to_sub_p = dict(np.stack((idx, sub_p), axis=1))
    return id_to_sub_p


def subsampling(sentence_, id_to_sub_p):  # sentence: list, id_to_sub_p : dictionary
    corpus = []
    for word in sentence_:
        # double check, if center word is '\s', skip
        random_num = np.random.random()
        if word == 0: continue
        if id_to_sub_p[word] > random_num:
            corpus.append(word)
    return corpus

def unigramsampler(vocab, word_to_id, power = 3/4):
    sample_table = []
    current = 0
    pos_idx = []
    for word in tqdm(vocab.keys(), desc='Unigram Sampling'):
        if word_to_id[word] == 0:
            continue
        freq = int(pow(vocab[word], 3/4))
        pos_idx.append((current, current+freq))
        current += freq
        temp = [word_to_id[word]] * freq
        sample_table.extend(temp)
    print('Unigram Table length: ', len(sample_table))
    return sample_table



def make_contexts(window_size, corpus, target_idx):
    window = int(np.random.randint(1, window_size+1))
    contexts = []
    for j in range(-window, window+1):  # window 가 2면 j= -2, -1, 0, 1, 2
        position = target_idx+j
        # target = 0 이면 contexts 는 0 다음부터만 생각해야됨, target은 contexts에 포함시키지 않음.
        if position >= 0 and position != target_idx:
            # else:
            #     contexts.append(corpus[position])         # complete contexts
            if position >= len(corpus):  # corpus 의 마지막을 넘어가면 안되고 for문도 더 돌 필요가 없음.
                break
            elif corpus[position] == 0: 
                continue
            else:
                contexts.append(corpus[position])

    return contexts


def count_total_word(sentence_list):
    total_word = 0
    for sentence in sentence_list:
        for _ in sentence:
            total_word += 1
    print("total training word: ", total_word)
    return total_word



def test2():
    corpus_list = [[0, 1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    for target in range(len(corpus_list[0])):
        if corpus_list[0][target] == 0:
            continue
        else: contexts = make_contexts(2, corpus_list[0], target)

        print("target, context: ", corpus_list[0][target],',', contexts)

# test2()