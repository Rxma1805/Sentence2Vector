import sys
sys.path.append('./src')
import data_io, params, SIF_embedding
import numpy as np

douban_cropus_path = '/bigdata/xiaoma/Assi12/douban.txt'
sentences = []
with open(douban_cropus_path) as f:
    for line in f:
        line = line.strip()
        line = line.split(':')[1]
        sentences.append(line)

glove_word2vector_path = './chinese_data_douban_cropus_vectors.txt' # word vector file, can be downloaded from GloVe website
word_freauency_path = './douban_cropus_vocab.txt' # each line is a word and its frequency
weightpara = 1e-3
rmpc = 1


# load word vectors
(Word2Indx, Word2vector) = data_io.getWordmap(glove_word2vector_path)
# load word weights
word2weight = data_io.getWordWeight(word_freauency_path, weightpara) # word2weight['str'] is the weight for the word 'str'
Index2Weight = data_io.getWeight(Word2Indx, word2weight) # weight4ind[i] is the weight for the i-th word



        
word_idx_seq_of_sentence, mask = data_io.sentences2idx(sentences, Word2Indx) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
word_weight_of_sentence = data_io.seq2weight(word_idx_seq_of_sentence, mask, Index2Weight) # get word weights

# set parameters
params = params.params()
params.rmpc = rmpc
embedding = SIF_embedding.SIF_embedding(Word2vector, word_idx_seq_of_sentence, word_weight_of_sentence, params)
np.save("douban_sentence2vector.npy",embedding)
