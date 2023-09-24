import pickle

data_path = 'data/1-billion-word/preprocessed/'

with open(data_path + 'corpus/corpus100.pkl', 'rb') as fr:
    sentence_list = pickle.load(fr)
    fr.close()


print(sentence_list[:10])