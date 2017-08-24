import sys
import nltk
import random
import numpy as np
from collections import *
import os

'''
procedure:
1. divideUserCorpus
2. concatUserCorpus
3. splitUserCorpus
4. wordCount
5. createInputXY
'''

def wordCount():
	users_corpus = np.load("../corpus/users_all.npy")
	u   = []
	for s in users_corpus:
	    u += s
	users_counter = Counter(list(u))
	uc = []
	oov_u = 0
	oov_b = 0
	for word in users_counter:
		if users_counter[word] > 2:
			uc.append(word)
		else:
			oov_u += 1
	users_corpus = list(set(uc))
	
	background_corpus = list(np.load("../corpus/background_200000.npy"))
	b = []
	for s in background_corpus:
		b += s 
	background_counter = Counter(b)
	background_word_list = []
	for word in background_counter:
		if background_counter[word] > 2:
			background_word_list.append(word)
		else:
			oov_b += 1
	total_word_set = set(background_word_list + users_corpus)
	print len(total_word_set)
	N_dict = dict()
	index = 1
	for word in total_word_set:
		if not word.isdigit():
			N_dict[word] = index
			index += 1
	N_dict["&number"] = index
	N_dict["&endofsentence"] = index+1
	N_dict["&oov"] = index+2
	print "len", len(N_dict)
	print oov_u, oov_b
	np.save(open("N_dict_%i.npy" % len(N_dict), "wb"), N_dict)	

def userWordCount():
	users = ['porngus', 'seo_consultant', 'mariolavandeira', 'sudhir_vashist', 'tubebutler', 'rob_madden', 'techflypaper', 'real_advice_', 'eproducts24', 'bruneifm', 'webarticlesinfo', 'webnex', 'apthelpdesk', 'midasjohn', 'fluffy_a_bunny', 'python0java', 'articlesjack', 'audiom', 'ruhanirabin', 'freelistsd', 'pizdaorg', 'tmcmemberfeed', 'iqhq', 'gasdieselprices', 'wpstudios', 'masug', 'itweeveetoprt', 'topgossip', 'dikky_heartiez', 'dominiquerdr', 'gambling_casino', 'tin_tuc', 'ustalib', 'dileepiscalling', 'yuppmarks', 'bperry921', 'msrdinc', 'charlieandsandy', 'cutebutpsycho', 'algebra_com', 'pharmandsdinc', 'discountomatic']
	for u in users:
		c = np.load("../corpus/user/%s/all.npy" % u)
		s = []
		for sent in c:
			s += sent
		counter = Counter(list(s))
		n = set(counter)
		N = dict()
		i = 1
		for w in n:
			N[w] = i
			i += 1
		np.save("Tweets/user_N/N_%s.npy" % u, N)

def divideUserCorpus():
	print "dividing user Corpus"
	corpus = np.load("user_corpus_all.npy").item()
	f_c = np.load("friend_corpus_all.npy").item()
	for user in corpus:
		np.save(open("../corpus/user/%s/friend.npy" % user, "wb"), np.asarray(f_c[user]))
		np.save(open("../corpus/user/%s/user.npy" % user, "wb"), np.asarray(corpus[user]))
		np.save("../corpus/user/%s/all.npy" % user, np.asarray(corpus[user]+f_c[user]))

def concatUserCorpus():
	print "concating user Corpus"
	corpus = np.load("user_corpus_all.npy").item()
	f_c = np.load("friend_corpus_all.npy").item()
	uc = np.asarray([])
	for u in corpus:
		uc = np.concatenate((uc, corpus[u], f_c[u]))
	np.save("../corpus/users_all.npy", uc)

def splitUserCorpus():
	print "splitting user corpus into Train, Valid, Test corpuses"
	users = ['porngus', 'seo_consultant', 'mariolavandeira', 'sudhir_vashist', 'tubebutler', 'rob_madden', 'techflypaper', 'real_advice_', 'eproducts24', 'bruneifm', 'webarticlesinfo', 'webnex', 'apthelpdesk', 'midasjohn', 'fluffy_a_bunny', 'python0java', 'articlesjack', 'audiom', 'ruhanirabin', 'freelistsd', 'pizdaorg', 'tmcmemberfeed', 'iqhq', 'gasdieselprices', 'wpstudios', 'masug', 'itweeveetoprt', 'topgossip', 'dikky_heartiez', 'dominiquerdr', 'gambling_casino', 'tin_tuc', 'ustalib', 'dileepiscalling', 'yuppmarks', 'bperry921', 'msrdinc', 'charlieandsandy', 'cutebutpsycho', 'algebra_com', 'pharmandsdinc', 'discountomatic']
	#users = ['algebra_com']
	C_TRAIN = np.asarray([])
	C_VALID = np.asarray([])
	C_TEST  = np.asarray([])
	F_TRAIN = np.asarray([])
	F_VALID = np.asarray([])
	F_TEST  = np.asarray([])
	for u in users:
		C = np.load("../corpus/user/%s/all.npy" % u)
		F = np.load("../data/feature/sem/update/%s_d2_p5.npy")
		index = range(len(C))
		n = len(C) / 5
		random.shuffle(index)
		C = C[index]
		F = F[index]
		C_train = C[:3*n]
		C_valid = C[3*n:4*n]
		C_test  = C[4*n:]
		F_train = F[:3*n]
		F_valid = F[3*n:4*n]
		F_test  = F[4*n:]
		C_TRAIN = np.concatenate((C_TRAIN, C_train))
		C_VALID = np.concatenate((C_VALID, C_valid))
		C_TEST  = np.concatenate((C_TEST , C_test ))
		F_TRAIN = np.concatenate((F_TRAIN, F_train))
		F_VALID = np.concatenate((F_VALID, F_valid))
		F_TEST  = np.concatenate((F_TEST , F_test ))
	C_TRAIN = sorted(C_TRAIN, key=lambda x:len(x))
	C_VALID = sorted(C_VALID, key=lambda x:len(x))
	C_TEST  = sorted(C_TEST,  key=lambda x:len(x))
	F_TRAIN = sorted(F_TRAIN, key=lambda x:len(x))
	F_VALID = sorted(F_VALID, key=lambda x:len(x))
	F_TEST  = sorted(F_TEST,  key=lambda x:len(x))
	
	print len(C_TRAIN), len(C_VALID), len(C_TEST)
	print len(F_TRAIN), len(F_VALID), len(F_TEST)
	np.save("../corpus/user/train.npy", C_TRAIN)
	np.save("../corpus/user/valid.npy", C_VALID)
	np.save("../corpus/user/test.npy" , C_TEST )
	np.save("../data/feature/sem/user/train.npy", F_TRAIN)
	np.save("../data/feature/sem/user/valid.npy", F_VALID)
	np.save("../data/feature/sem/user/test.npy" , F_TEST )

def createInputXY(): # create user / friend input
	c_0 = np.load("../corpus/user/train.npy")
	c_1 = np.load("../corpus/user/valid.npy")
	c_2 = np.load("../corpus/user/test.npy")
	c_3 = np.load("../corpus/background_200000.npy")
	N_dict = np.load("N_dict_33652.npy").item()
	C = [c_0, c_1, c_2, c_3]
	C_X, C_Y = [], []
	for c in C:
		X = []
		Y = []
		N = len(N_dict)
		for sentence in c:	
			n_words = len(sentence)
			idx_list = np.zeros(n_words+1)
			x = []
			y = []
			idx_list[-1] = N-2             #last word in sentence
			for i in range(n_words):
				index = 0
				if sentence[i].isdigit():   # check wheather token contains number or not
					index = N-3 	    	# if yes, arrange to the last index(not OOV)
				else:
					index = N_dict.get(sentence[i],N-1)
				idx_list[i] = index
			if 0 in idx_list:
				print "WARN!!!"
			x = idx_list[:-1]
			y = idx_list[1:]
			X.append(x)
			Y.append(y)
		X = np.asarray(X)
		Y = np.asarray(Y)
		C_X.append(X)
		C_Y.append(Y)
	print len(C_X[0]), len(C_X[1]), len(C_X[2]), len(C_X[3])
	np.save("../data/user/X_train.npy", C_X[0])
	np.save("../data/user/Y_train.npy", C_Y[0])
	
	np.save("../data/user/X_valid.npy", C_X[1])
	np.save("../data/user/Y_valid.npy", C_Y[1])
	
	np.save("../data/user/X_test.npy", C_X[2])
	np.save("../data/user/Y_test.npy", C_Y[2])
	
	np.save("../data/X_train_200000.npy", C_X[3])
	np.save("../data/Y_train_200000.npy", C_Y[3])	

#divideUserCorpus()
#concatUserCorpus()
wordCount()
#splitUserCorpus()
#userWordCount()
#createInputXY()
