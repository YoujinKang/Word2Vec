import numpy as np
import argparse
import pickle
import os
import sys
sys.path.append(os.getcwd())

def check_valid(questions, vocab):
    valid_question = []
    for question in questions:  # questions 안의 문장에는 각 4개의 단어들이 나열되어있음.
        if question[0] in vocab and question[1] in vocab and question[2] in vocab and question[3] in vocab:
            valid_question.append(question)  # 한 문장의 모든 단어가 vocabulary에 있을 때만 유효문제로 넘김.
    return valid_question


def convert_to_vec(valid, word_vectors, word_to_id):  
    vec0 = []
    vec1 = []
    vec2 = []
    vec3 = []
    for s in valid:  # 유효 문제 리스트에서 한문장씩
        vec_temp0 = word_vectors[word_to_id[s[0]]]  # 문장 내 한 단어씩 룩업
        vec_temp1 = word_vectors[word_to_id[s[1]]]
        vec_temp2 = word_vectors[word_to_id[s[2]]]
        vec_temp3 = word_vectors[word_to_id[s[3]]]

        norm0 = np.linalg.norm(vec_temp0)  # 벡터의 norm을 구해서
        norm1 = np.linalg.norm(vec_temp1)
        norm2 = np.linalg.norm(vec_temp2)
        norm3 = np.linalg.norm(vec_temp3)

        vec0.append(vec_temp0 / norm0)  # normalize해서 넣어줌
        vec1.append(vec_temp1 / norm1)
        vec2.append(vec_temp2 / norm2)
        vec3.append(vec_temp3 / norm3)
    # vec0은 0번째 단어들에 대한 벡터 모임, vec1은 1번째 단어들에 대한 벡터모임 등등
    return np.array(vec0), np.array(vec1), np.array(vec2), np.array(vec3)
    # numpy array로 반환

def cos_similarity(predict, word_vectors):
    norm_predict = np.linalg.norm(predict, axis=1)  # predict 의 1번 axis에서 norm 계산 (제곱의 sqrt)
    norm_words = np.linalg.norm(word_vectors, axis=1)  

    similarity = np.dot(predict, word_vectors.T)  # similarity = (N, V) 차원
    similarity *= 1 / (np.add(norm_words,1e-8))
    similarity = similarity.T
    similarity *= 1 / (np.add(norm_predict,1e-8))  # 1e-8은 0으로 나눠지는 것을 방지하기 위한 작은 값
    similarity = similarity.T

    return similarity


def count_in_top4(similarity, id_to_word, valid):
    count = 0
    top4_idx = []
    top4_sim = []
    for i in range(len(similarity)):  
        sort_idx = np.argsort(similarity[i])  # similarity[i] 의 오름차순 순으로 idx 리스트 저장
        max_arg = sort_idx[::-1]  # 어레이를 뒤집은 index -> 내림차순에대한 idx
        temp = list(max_arg[:4])  # similarity[i]중 큰 순서에 대한 idx 4개만 리스트로 만듦
        top4_idx.append(temp)  # top idx 4씩 리스트로 담긴다. [[0,2,1,3], [9,22,11,12],..., [...]]
        top4_sim.append(list(similarity[i][temp]))  # top idx 4개에 대한 similarity를 리스트형태로 담는다.

        for j in range(4):  # temp에 담긴 4개의 단어를 하나씩 훑는다
            pred = id_to_word[temp[j]]  # temp에 담긴 id에 해당되는 word 찾음
            if pred in valid[i]:  # 그 단어가 문장 내에 있어야하고
                if pred == valid[i][3]:  # 그중에서도 해당 문장 내 마지막단어에 해당되면 카운트
                    count += 1
            else:
                break  # valid 에 없으면 무시
    return top4_idx, top4_sim, count



parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str, default='./CBOW_HS/train_log/CBOW_HS_sub1e-05_window8_dim300_1epoch/weight')
parser.add_argument('--weight', type=str, default='ckpt_weight.pkl')
parser.add_argument('--in_or_out', type=str, default='in')
parser.add_argument('--data_seg', type=int, default=10)

args = parser.parse_args()

# load analogy data
with open("data/1-billion-word/questions-words.txt", 'r') as fr:
    loaded = fr.readlines()
count = 0
semantic = []
syntactic = []
for line in loaded:
    if line[0] == ':':  # 새로운 유형의 question시작
        count += 1  
        continue
    elif line == '\n':  # 다음 question
        continue
    if count < 6:  # 총 16개의 유형 중 앞 5개까지는 의미질문 6번째부터 마지막까지는 문법질문
        semantic.append(line.split())
    else:
        syntactic.append(line.split())

# load word vectors
file = os.path.join(args.log_dir, args.weight)
with open(file, 'rb') as fr:
    vectors = pickle.load(fr)
word_vectors = None
if args.in_or_out == 'in':  # W_in
    word_vectors = np.array(vectors[0])
elif args.in_or_out == 'out':  # W_out
    word_vectors = np.array(vectors[1])

# load dictionary
with open("data/1-billion-word/preprocessed/dictionary_all.pkl", 'rb') as fr:
    word_to_id, id_to_word, vocab = pickle.load(fr)

# Check whether my word vectors contain all words in questions
valid_sem = check_valid(semantic, vocab)  # 문장의 단어들이 vocab에 있는 것만 선택
valid_syn = check_valid(syntactic, vocab)
print("valid semantic: %d/%d" %(len(valid_sem), len(semantic))) # 유효한 갯수 출력
print("valid syntactic: %d/%d" %(len(valid_syn), len(syntactic)))


#### evaluation시작####

batch1 = len(valid_syn)//args.data_seg + 1  
batch2 = len(valid_sem)//args.data_seg + 1
# 예) 유효질문지가 57개면 batch = 5+1 = 6
# 0-6, 6-12, 12-18, ..., 48-54, 54-60 인데 뒷부분 갯수 모자라면 끊김. 54-57로
syn_counts = 0
sem_counts = 0
for i in range(args.data_seg):
    print('Checking data: {}/{}'.format(i+1, args.data_seg))
    batch_syn = valid_syn[i*batch1: (i+1)*batch1] # 유효 질문 목록에서 배치사이즈만큼만 가져옴
    batch_sem = valid_sem[i*batch2: (i+1)*batch2]

    # syntactic
    a1, b1, c1, d1 = convert_to_vec(batch_syn, word_vectors, word_to_id)
    # a1 에는 각 문장의 첫 단어들만 모임, b1은 두번째 단어들만 모임.. 등등
    predict_syn = b1 - a1 + c1  
    similarity_syn = cos_similarity(predict_syn, word_vectors)
    # b-a+c 한 벡터와 W_in 의 cos similarity 구함
    syn_max_top4, syn_sim_top4, syn_count = count_in_top4(similarity_syn, id_to_word, batch_syn)
    # syn_count는 batch_syn을 한줄씩 읽을 때 각 줄의 마지막 단어가 cos similarity내의 top4에 있는 갯수
    syn_counts += syn_count

    #semantic도 마찬가지
    a2, b2, c2, d2 = convert_to_vec(batch_sem, word_vectors, word_to_id)
    predict_sem = b2 - a2 + c2
    similarity_sem = cos_similarity(predict_sem, word_vectors)
    sem_max_top4, sem_sim_top4, sem_count = count_in_top4(similarity_sem, id_to_word, batch_sem)
    sem_counts += sem_count

syn_acc = syn_counts/len(valid_syn) * 100
sem_acc = sem_counts/len(valid_sem) * 100
print("syntactic accuracy: ", syn_acc)
print("semantic accuracy: ", sem_acc)
print("total accuracy: ", (syn_acc*len(valid_syn) + sem_acc*len(valid_sem))/(len(valid_syn)+len(valid_sem)))
