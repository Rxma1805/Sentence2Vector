from SIF import data_io, params, SIF_embedding
glove_file = 'vectors.txt'

word_freauency_path = 'vocab.txt'
weight_params = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme

(Word2Indx, Word2vector) = data_io.getWordmap(glove_file)    
Word2Weight = data_io.getWordWeight(word_freauency_path, weight_params) 
Index2Weight= data_io.getWeight(Word2Indx, Word2Weight) 

def embeding_sentence_cosine_similarity(s1,s2):    
    word_idx_seq_of_sentence, mask = data_io.sentences2idx([s1,s2], Word2Indx) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    print(s1,s2)
    print('word_idx_seq_of_sentence')
    print(word_idx_seq_of_sentence)
    print('mask')
    print(mask)
    word_weight_of_sentence = data_io.seq2weight(word_idx_seq_of_sentence, mask, Index2Weight) # get word weights
    # set parameters
    param = params.params()
    param.rmpc = rmpc
    embedding = SIF_embedding.SIF_embedding(Word2vector, word_idx_seq_of_sentence, word_weight_of_sentence, param) 
    s1_embed = embedding[0]
    s2_embed = embedding[1]    

    return distance.cosine(s1_embed,s2_embed)
