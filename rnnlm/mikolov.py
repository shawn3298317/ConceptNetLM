from keras.models import Sequential
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
import numpy as np
import argparse
import os
from math import e
from numpy.lib.scimath import logn

import time
import sys
from utils import *

'M_PARAMATERS = [FEATURE_ADD,]'

#TRAINING PARAMTERS DEFINE
EPOCH = 13
BATCH = 25
DATA_LENGTH = 10000#10000
VAL_LENGTH  = 999
MAX_LEN = 100
N = 2291#40401#44573#103192
FEAT_SIZE = 48

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

def evaluate_ppl(y_t_batch, y_h_batch):
	y_t_batch = np.clip(y_t_batch, 1e-20, 1-1e-20)
	total_loss = []
	for n in range(len(y_t_batch)):
		H = 0.
		for t in range(len(y_t_batch[n])):
			if 1 in y_h_batch[n][t]:
				index = np.where(y_h_batch[n][t] == 1)[0][0]
				H += logn(10, y_t_batch[n][t][index])
				#H += np.sum(np.multiply(y_h_batch[n][t], logn(2,y_t_batch[n][t])))
			else:
				loss = (H/t)
				break
		total_loss.append(loss)
	loss = -sum(total_loss) / len(total_loss)
	return loss

def flush_print(t, loss, acc, total, time1, time2, ep):

	toWrite = "\rEp:%i %i/%i ppl:%f, acc:%f, per_batch:%.3fs, load_time:%.3fs " % (ep+1,t,total,loss,loss2,acc,time1,time2)
	toWrite += calc((total-t) * (time2+time1))
	sys.stdout.write(toWrite)
	sys.stdout.flush()


def init_model():
		WORD_SIZE = 1#2721127
		WORD_EMBED = 300#300

		word = Sequential()
		word.add(Embedding(input_dim=N, input_length=MAX_LEN, output_dim=WORD_EMBED, mask_zero=True))

		feature = Sequential()
		feature.add(LSTM(output_dim = FEAT_SIZE, input_shape=(MAX_LEN, FEAT_SIZE), activation='sigmoid',return_sequences=True))
		
		model = Sequential()
		model.add(Merge([word,feature], mode='concat'))
		model.add(LSTM(output_dim = 500, input_shape=(MAX_LEN, FEAT_SIZE+WORD_EMBED), activation='sigmoid',return_sequences=True)) 
		model.add(Dropout(0.3))
		model.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=500, activation='softmax'))
		model.add(Dropout(0.3))
		
		print "Compiling"
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		return model

parser = argparse.ArgumentParser()
parser.add_argument('-feat', type=str)
args = parser.parse_args()
print "Training with %s feature" % args.feat
	
print "Loading data...."
X_valid = np.load("../mikolov/X_valid.npy")
X_train = np.load("../mikolov/X_train.npy")
Y_train = np.load("../mikolov/Y_train.npy")
Y_valid = np.load("../mikolov/Y_valid.npy")
if args.feat != None:
	F_train = np.load("../data/background/F_%s_train.npy" % args.feat)
	F_valid = np.load("../data/background/F_%s_valid.npy" % args.feat)

print "Initializing..."
model = init_model()


valid_ppl = []
for ep in range(EPOCH):
	
	total_loss1 = 0.0
	total_acc1 = 0.0
	total_loss2 = 0.0
	total_acc2 = 0.0
	for t in range(DATA_LENGTH/BATCH):

		time_st = time.time()
		
		x    = X_train[BATCH*t:BATCH*(t+1)]
		y    = Y_train[BATCH*t:BATCH*(t+1)]
		if args.feat != None:
			feat = F_train[BATCH*t:BATCH*(t+1)]
		
		timesteps = MAX_LEN#len(x[-1])
		#print timesteps # sentences sorted in ascending order of length(0,...,30)
		new_X    = np.zeros((BATCH,timesteps)) # time series sentence
		new_Y    = np.zeros((BATCH,timesteps,N))
		new_FEAT = np.zeros((BATCH,timesteps,FEAT_SIZE))
		for n in range(BATCH):
			for i in range(len(y[n])):
				idx = y[n][i]
				new_Y[n][i][idx] = 1.0
				new_X[n][i] = x[n][i]
				if args.feat != None:
					new_FEAT[n][i] = np.asarray(feat[n][i])

		time_md2 = time.time()
		#print 'X_batch shape:', new_X.shape
		#print 'Y_batch shape:', new_Y.shape
		#print 'F_batch shape:', new_FEAT.shape
		loss1, acc1 =  model.train_on_batch([new_X, new_FEAT], new_Y, accuracy=True, class_weight=None, sample_weight=None)

		total_loss1 += loss1
		total_acc1 += acc1
		
		time_en = time.time()
		flush_print(t+1,e**(total_loss1/(t+1)),total_acc1/(t+1),DATA_LENGTH/BATCH, time_en-time_md2, time_md2-time_st, ep)

	for t2 in range(VAL_LENGTH/BATCH):
		x = X_valid[BATCH*t2:BATCH*(t2+1)]
		y = Y_valid[BATCH*t2:BATCH*(t2+1)]
		if args.feat != None:
			feat = F_valid[BATCH*t2:BATCH*(t2+1)]
		
		timesteps = MAX_LEN#len(x[-1])
		
		new_X = np.zeros((BATCH,timesteps))
		new_Y = np.zeros((BATCH,timesteps,N))
		new_FEAT = np.zeros((BATCH,timesteps,FEAT_SIZE))

		for n in range(BATCH):
			for i in range(len(y[n])):
				idx = y[n][i]
				new_Y[n][i][idx] = 1.0
				new_X[n][i] = x[n][i]
				if args.feat != None:
					new_FEAT[n][i] = np.asarray(feat[n][i])

		#loss2, acc2 = model.test_on_batch([new_X,new_FEAT], new_Y, accuracy=True, sample_weight=None)
		y_pdct = model.predict([new_X, new_FEAT], batch_size=BATCH)
		loss2 = evaluate_ppl(y_pdct, new_Y)
		#y_pdct = np.clip(y_pdct, 1e-20, 1 - 1e-20)
		#y_pdct /= y_pdct.sum(axis=1)[:, np.newaxis]
		#loss2 = -(new_Y * np.log(y_pdct)).sum()
	
		total_loss2 += loss2
		#flush_print(t2+1,total_loss2/(t2+1),total_acc2/(t2+1),DATA_LENGTH/BATCH, time_en-time_md2, time_md2-time_st, ep)

	ppl = total_loss2/(t2+1)
	valid_ppl.append(ppl)
	

	sys.stdout.write( "\rEp:%i %i/%i train_loss:%f, train_acc:%f, val_loss:%f, val_ppl:%f                                                   \
		\n" % (ep+1,t+1,DATA_LENGTH/BATCH,total_loss1/(t+1),total_acc1/(t+1),total_loss2/(t2+1)))
	#m_json = model.to_json()
	#open('model/LM_%s_%i_%i.json' % (args.feat, ep, FEAT_SIZE), 'w+').write(m_json)
	model.save_weights('weight/LM_%s_%i_%i.h5' % (args.feat, ep, FEAT_SIZE), overwrite=True)

index = valid_ppl.index(min(valid_ppl))
for i in range(13):
	if i != index:
		os.remove('weight/LM_%s_%s_%i.h5' % (args.feat, i, FEAT_SIZE))
	else:
		os.rename('weight/LM_%s_%s_%i.h5' % (args.feat, i, FEAT_SIZE), 'weight/saved/LM_%s.h5' % args.feat)


#TODO LIST:
#1. Train with word only
#2. Add features to training model
#3. Construct update platform
#4. TF-IDF for user-corpus
