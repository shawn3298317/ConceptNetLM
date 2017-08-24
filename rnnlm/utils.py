import random
import numpy as np
import sys
import time
from numpy.lib.scimath import logn

import theano
import theano.tensor as T

def makeBackgroundData():
	X = np.load("../SocialNetwork/X.npy")
	Y = np.load("../SocialNetwork/Y.npy")
	Fp = np.load("../feat_extract/POS_Feature/background_pos_feat.npy")
	Fs = np.load("../feat_extract/POS_Feature/background_sem_feat.npy")
	num = 2500
	totalLen = 30000 
	index = sorted(random.sample(range(0, totalLen), num))
	X_test = []
	Y_test = []
	F_testp = []
	F_tests = []
	for i in index:
		X_test.append(X[i])
		Y_test.append(Y[i])
		F_testp.append(Fp[i])
		F_tests.append(Fs[i])
	X = np.delete(X, index)
	Y = np.delete(Y, index)
	Fp = np.delete(Fp, index)
	Fs = np.delete(Fs, index)
	np.save("../data/background/X_train.npy", X)
	np.save("../data/background/X_valid.npy", X_test)
	np.save("../data/background/Y_train.npy", Y)
	np.save("../data/background/Y_valid.npy", Y_test)
	np.save("../data/background/F_pos_train.npy", Fp)
	np.save("../data/background/F_pos_valid.npy", F_testp)
	np.save("../data/background/F_sem_train.npy", Fs)
	np.save("../data/background/F_sem_valid.npy", F_tests)
	#return X, np.asarray(X_test), Y, np.asarray(Y_test), Fp, np.asarray(F_testp), Fs, np.asarray(F_tests)

def calculatePPL(result):
	train_non_ppl = []
	test_non_ppl  = []
	train_pos_ppl = []
	test_pos_ppl  = []
	train_sem_ppl = []
	test_sem_ppl  = []
	with open(result, 'r') as f:
		for line in f:
			if line.split(",")[2][-3:] == "one":
				train_non_ppl.append(float(line.split(",")[-1].split(":")[-1]))
				test_non_ppl.append(float(line.split(",")[-2].split(":")[-1]))
			if line.split(",")[2][-3:] == "pos":
				train_pos_ppl.append(float(line.split(",")[-1].split(":")[-1]))
				test_pos_ppl.append(float(line.split(",")[-2].split(":")[-1]))
			if line.split(",")[2][-3:] == "sem":
				train_sem_ppl.append(float(line.split(",")[-1].split(":")[-1]))
				test_sem_ppl.append(float(line.split(",")[-2].split(":")[-1]))
		
	print "Train none ppl:", len(train_non_ppl)
	print sum(train_non_ppl) / len(train_non_ppl)
	print "Test none ppl:"
	print sum(test_non_ppl) / len(test_non_ppl)
	print "Train pos ppl:", len(train_pos_ppl)
	print sum(train_pos_ppl) / len(train_pos_ppl)
	print "Test pos ppl:" 
	print sum(test_pos_ppl) / len(test_pos_ppl)	
	print "Train sem ppl:", len(train_sem_ppl)
	print sum(train_sem_ppl) / len(train_sem_ppl)
	print "Test sem ppl:" 
	print sum(test_sem_ppl) / len(test_sem_ppl)
#calculatePPL('results.txt')

def splitUserCorpus(user):
	C = np.load('../corpus/user/%s/all.npy' % user)
	index = range(1000)	
	random.shuffle(index)
	C = C[index]
	C_train = C[:600]
	C_valid = C[600:800]
	C_test  = C[800:]
	C_train = sorted(C_train, key=lambda x:len(x))
	C_valid = sorted(C_valid, key=lambda x:len(x))
	C_test  = sorted(C_test,  key=lambda x:len(x))
	
	#	return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, F_train, F_valid, F_test
	#else:
	#	return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
	np.save("../corpus/user/%s/train.npy" % user, C_train)
	np.save("../corpus/user/%s/valid.npy" % user, C_valid)
	np.save("../corpus/user/%s/test.npy"  % user,  C_test)
	
#users = ['topofstuff', 'nieuwslogr', 'porngus', 'yyz1', 'site_news', 'seo_consultant', 'ocbarista', 'hotnews26', 'lingonews', 'flying_gramma', 'gluetext', 'ocnighter', 'topnews25', 'pflive_announce', 'annuitypayments', 'hot_bp', 'bestofyoutubers', 'delicious50', 'financetip', 'articlescreek', 'triplekillsblog', 'articles_mass', 'articlesmob', 'jobhits', 'mariolavandeira', 'sudhir_vashist', 'tubebutler', 'getfreelancejob', 'spitzezeug', 'rob_madden', 'car__tips', 'fooshare', 'blocalbargains', 'valuescompanies', 'techflypaper', 'articles4author', 'newsweb2x', 'real_advice_', 'mumbaitimes', 'newstop_us', 'dragtotop', 'tiptop_trends_1']
#for u in users:
#	print u
#	splitUserCorpus(u)

def loadUserData(user, feat):
	if feat != None:
		X_train = np.load("../data/train/X/%s.npy" % user)
		X_valid	= np.load("../data/valid/X/%s.npy" % user)
		X_test  = np.load("../data/test/X/%s.npy" % user)
		Y_train = np.load("../data/train/Y/%s.npy" % user)
		Y_valid = np.load("../data/valid/Y/%s.npy" % user)
		Y_test  = np.load("../data/test/Y/%s.npy" % user)
		F_train = np.load("../data/train/%s/%s.npy" % (feat,user)) 
		F_valid = np.load("../data/valid/%s/%s.npy" % (feat,user))
		F_test  = np.load("../data/test/%s/%s.npy" % (feat,user))
		return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, F_train, F_valid, F_test
	else:
		X_train = np.load("../data/train/X/%s.npy" % user)
		X_valid	= np.load("../data/valid/X/%s.npy" % user)
		X_test  = np.load("../data/test/X/%s.npy" % user)
		Y_train = np.load("../data/train/Y/%s.npy" % user)
		Y_valid = np.load("../data/valid/Y/%s.npy" % user)
		Y_test  = np.load("../data/test/Y/%s.npy" % user)
		return X_train, X_valid, X_test, Y_train, Y_valid, Y_test	

def loadBatch(X, Y, F, t, MAX_LEN, BATCH, N, FEAT_SIZE, args_feat):
	x = X[BATCH*t:BATCH*(t+1)]
	y = Y[BATCH*t:BATCH*(t+1)]
	if args_feat != None:
		feat = F[BATCH*t:BATCH*(t+1)]
	
	timesteps = MAX_LEN#len(x[-1])
	
	new_X = np.zeros((BATCH,timesteps)) # time series sentence
	new_Y = np.zeros((BATCH,timesteps,N))
	new_F = np.zeros((BATCH,timesteps,FEAT_SIZE))
	length = 0.0
	for n in range(BATCH):
		for i in range(len(y[n])):
			idx = int(y[n][i]) - 1
			length += 1
			new_Y[n][i][idx] = 1.0
			new_X[n][i] = x[n][i]
			if  args_feat != None:
				new_F[n][i] = np.asarray(feat[n][i])
	return new_X, new_Y, new_F, length 

def evaluate_ppl(y_t_batch, y_h_batch):
	y_t_batch = np.clip(y_t_batch, 1e-20, 1-1e-20)
	total_loss = []
	for n in range(len(y_t_batch)):
		H = 0.
		for t in range(len(y_t_batch[n])):
			if 1 in y_h_batch[n][t]:
				index = np.where(y_h_batch[n][t] == 1)[0][0]
				H += logn(2, y_t_batch[n][t][index])
				#H += np.sum(np.multiply(y_h_batch[n][t], logn(2,y_t_batch[n][t])))
			else:
				loss = (H/t)
				break
		total_loss.append(loss)
	loss = -sum(total_loss) / len(total_loss)
	return loss

def evaluate_ppl_2():

	t_dist = T.tensor3()
	c_dist = T.tensor3()

	c_clip = T.clip(c_dist, 1e-15, 1)
	ret = theano.tensor.nnet.categorical_crossentropy(c_clip,t_dist)
	#ret = T.sum(t_dist * T.log(c_clip),axis=-1)
	#func = T.sum(T.sum(ret,axis=-1),axis=-1)
	func = T.sum(ret)
	f = theano.function([c_dist,t_dist],func,allow_input_downcast=True)

	return f

def calc(eta):
	day = 0
	hrs = 0
	mins = 0
	sec = 0.0
	day = eta / 86400
	hrs = (eta % 86400) / 3600
	mins = ((eta % 86400) % 3600) / 60
	sec = ((eta % 86400) % 3600) % 60

	ret = "ETA:%ih %im %is " % (hrs,mins,sec)
	return ret

def flush_print(t, loss, acc, total, time1, time2, ep):
	toWrite = "\rEp:%i %i/%i loss:%f, acc:%f, per_batch:%.3fs, load_time:%.3fs " % (ep+1,t,total,loss,acc,time1,time2)
	toWrite += calc((total-t) * (time2+time1))
	sys.stdout.write(toWrite)
	sys.stdout.flush()

