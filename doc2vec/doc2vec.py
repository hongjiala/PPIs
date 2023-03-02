# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:19:53 2022

@author: xiepengfei
"""

import numpy as np
from Bio import SeqIO
from nltk import trigrams, bigrams,ngrams
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

np.set_printoptions(threshold=np.inf)

names = ["DENV","Hepatitis","Herpes","HIV","Influenza","Papilloma","SARS2","ZIKV"]
for name in names:
    texts = []
    for index, record in enumerate(SeqIO.parse('fasta/%s.fasta'%name, 'fasta')):
        tri_tokens = ngrams(record.seq,6)
        temp_str = ""
        for item in ((tri_tokens)):
            # print(item),

            items = ""
            for strs in item:
                items = items+strs
            temp_str = temp_str + " " + items
            #temp_str = temp_str + " " +item[0]
        texts.append(temp_str)



    seq=[]
    stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    for doc in texts:
        doc = re.sub(stop, '', doc)
        seq.append(doc.split())

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(seq)]
    model = Doc2Vec(documents , vector_size=1000, window=500, min_count=1, workers=12)
    model.train(documents ,total_examples=model.corpus_count, epochs=50)
    #model.save("autodl-tmp/my_doc2vec_model.model") # you can continue training with the loaded model!
    #model.dv.save_word2vec_format('%s.vector'%name)

    # test_seq = ['MRQGCKFRGSSQKIRWSRSPPSSLLHTLRPRLLSAEITLQTNLPLQSPCCRLCFLRGTQAKTLK']
    # # test_text = ngrams(test_seq,6)
    # # temp_str_test = ""
    # # for item in ((test_text)):
    # #         # print(item),
    # #     print(item)
    # #     items = ""
    # #     for strs in item:
    # #         items = items+strs
    # #     temp_str = temp_str_test + " " + items
    # inferred_vector_dm = model.infer_vector(test_seq)
    # print(inferred_vector_dm)
    np.save("vec/new_%s_vector.npy"%name,model.dv.vectors)