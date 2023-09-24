from collections import Counter
import re
from tqdm.auto import tqdm
import os
import sys
sys.path.append(os.getcwd())

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# def make_words(file):
#     word_list = []
#     with open(file, 'r', encoding='utf8') as fr:
#         data = fr.readlines()
#         fr.close()
#     sent_text = [clean_str(sent) for sent in data]
#     for line in sent_text:
#         word_list.extend(line.split())
#         word_list.append('<\s>')  #문장 끝에 토큰
#     return word_list, sent_text

def make_words(file):
    with open(file, 'r', encoding='utf-8') as fr:
        data = fr.read()

    word_list = []
    word = ''
    for c in data:
        if c in ['\n', ' ', '\t']:
            word_list.append(word)
            word = ''
            if c == '\n':
                word_list.append('<\s>')
        else: word += c
    return word_list

def make_dict(start, end, min_count=5):
    counter = Counter()
    for i in tqdm(range(start, end+1)):
        if i < 10:
            file = './data/1-billion-word/training/news.en-0000' + str(i) + '-of-00100'
        else:
            file = './data/1-billion-word/training/news.en-000' + str(i) + '-of-00100'
        # word_list, _ = make_words(file)
        word_list = make_words(file)
        counter += Counter(word_list)
    word_list = make_words(file)
    counter += Counter(word_list)

    temp = Counter()
    for word in counter.keys():
        if counter[word] < min_count:
            temp[word] += 100  # 무조건 없어지게 min_count보다 훨씬 큰 값을 할당

    counter -= temp
    vocabulary = {'<\s>': 0}
    vocabulary.update(dict(counter.most_common()))

    word_to_id = {}
    id_to_word = {}
    for word in vocabulary.keys():
        word_to_id[word] = len(word_to_id)
        id_to_word[len(id_to_word)] = word

    return word_to_id, id_to_word, vocabulary


# def make_sentences(word_to_id, file_num):
    
#     if file_num < 10:
#         file = f'./data/1-billion-word/training/news.en-0000{file_num}-of-00100'
#     else:
#         file = f'./data/1-billion-word/training/news.en-000{file_num}-of-00100'
#     _, sent_text = make_words(file)

#     sentence_list = []
#     for line in sent_text:

#         sentences = []
#         for word in line.split():
#             if word in word_to_id.keys():
#                 word_id = word_to_id[word]
#             else:
#                 word_id = 0
#             sentences.append(word_id)

#         sentence_list.append(sentences)

#     return sentence_list


def make_sentences(word_to_id, file_num):
    if file_num < 10:
        file = f'./data/1-billion-word/training/news.en-0000{file_num}-of-00100'
    else:
        file = f'./data/1-billion-word/training/news.en-000{file_num}-of-00100'
    word_list = make_words(file)

    sentences = []
    sentence_list = []
    for word in word_list:
        if word in word_to_id.keys():
            word_id = word_to_id[word]
            if word == '<\s>':
                sentence_list.append(sentences)
                sentences = []
            else: sentences.append(word_id)
    return sentence_list




def test2():
    words_list = make_words('./data/1-billion-word/training/news.en-00000-of-00100')
    print(words_list[:20])
    print('length: ', len(words_list))

    # word_to_id, id_to_word, vocab = make_dict(0, 0, min_count=0, freq_num=1000000)
    # print(word_to_id['<\s>'], id_to_word[0])
    # print(len(vocab))
    # time0 = time.time()
    # sentence_list = make_sentences(word_to_id, 99)
    # print(sentence_list[:2])
    # print(len(sentence_list))
    # print('time ; ', time.time()-time0)

# test2()

