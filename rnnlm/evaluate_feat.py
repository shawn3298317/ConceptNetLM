from keras.models import Sequential
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
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
parser = argparse.ArgumentParser()
parser.add_argument('-feat', type=str)
parser.add_argument('-data', type=str)
parser.add_argument('-d', type=int)
parser.add_argument('-p', type=int)
parser.add_argument('-weight', type=str)
args = parser.parse_args()
print "Testing with %s feature" % (args.data,args.feat)

#if args.data == "B+S":
DATA_LENGTH = 200000+25200
	#X_train = np.asarray(list(np.load("../data/X_train_200000.npy")) + list(np.load("../data/user/X_train.npy")))
	#Y_train = np.asarray(list(np.load("../data/Y_train_200000.npy")) + list(np.load("../data/user/Y_train.npy")))
#elif args.data == "B":
#	DATA_LENGTH = 200000
#	X_train = np.asarray(list(np.load("../data/X_train_200000.npy")))
	#X_train = np.asarray(list(np.load("../data/user/X_train.npy")))
#	Y_train = np.asarray(list(np.load("../data/Y_train_200000.npy")))
	#Y_train = np.asarray(list(np.load("../data/user/Y_train.npy")))

print "Loading data...."
X_valid = np.load("../data/user/X_test.npy")
Y_valid = np.load("../data/user/Y_test.npy")

feat_name = ""
if args.feat == 'pos':
	feat_name = 'pos'
        F_train = np.asarray(list(np.load("../data/feature/%s/naive_background.npy" % args.feat))+list(np.load("../data/feature/%s/naive_train.npy" % args.feat)))
        F_valid = np.load("../data/feature/%s/naive_valid.npy" % args.feat)
elif args.feat == 'sem':
	feat_name = 'sem_d%s_p%s' % (args.d, args.p)
        F_train = np.asarray(list(np.load("../data/feature/sem/background_naive_d%i_p%i.npy" % (args.d, args.p)))+list(np.load("../data/feature/sem/train_naive_d%i_p%i.npy" % (args.d, args.p))))
        F_valid = np.load("../data/feature/sem/valid_naive_d%i_p%i.npy" % (args.d, args.p))
elif args.feat == "w2v":
	F_train = np.asarray(list(np.load("../data/feature/w2v/background.npy"))+list(np.load("../data/feature/w2v/train.npy")))
	F_valid = np.load("../data/feature/w2v/valid.npy")
else:
	F_train, F_valid = [], []


EPOCH = 8
BATCH = 20
#DATA_LENGTH = 30000+25200
VAL_LENGTH  = 8400
MAX_LEN = 30
N = 33653#30225#44573#103192
FEAT_SIZE = 96

def init_model(weights_path):
		WORD_SIZE = 1#2721127
		WORD_EMBED = 300

		word = Sequential()
		word.add(Embedding(input_dim=N, input_length=MAX_LEN, output_dim=WORD_EMBED))
		if args.feat != None:
			feature = Sequential()
			feature.add(GRU(output_dim = FEAT_SIZE, input_shape=(MAX_LEN, FEAT_SIZE), activation='linear',return_sequences=True))
			#feature.add(Reshape(input_shape=(MAX_LEN, FEAT_SIZE), target_shape=(MAX_LEN, FEAT_SIZE)))		

			model = Sequential()
			model.add(Merge([word,feature], mode='concat'))
			model.add(GRU(output_dim = 300, input_shape=(MAX_LEN, WORD_EMBED+FEAT_SIZE), activation='linear',return_sequences=True)) 
			#model.add(Dropout(0.3))
			#model.add(LSTM(output_dim = 500, input_shape=(MAX_LEN, WORD_EMBED), activation='sigmoid',return_sequences=True))
			model.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=300, activation='softmax'))
			#model.add(Dense(output_dim=N,input_dim=500,activation='sigmoid'))
			#model.add(Dropout(0.3))
		
			print "Compiling with feat"
			model.load_weights(weights_path)
			model.compile(loss='categorical_crossentropy', optimizer='adam')
			return model
		else:
			word.add(GRU(output_dim=300, input_shape=(MAX_LEN, WORD_EMBED), activation='linear', return_sequences=True))
			word.add(TimeDistributedDense(output_dim=N, input_length=MAX_LEN, input_dim=300, activation='softmax'))
			print "Compiling without feat"
			word.compile(loss='categorical_crossentropy', optimizer='adam')
			return word	


print "Initializing..."
model = init_model("weight/" + args.weight)
valid_ppl = []
f_eval = evaluate_ppl_2()


for t2 in range(VAL_LENGTH/BATCH):
	new_X, new_Y, new_F, length = loadBatch(X_valid, Y_valid, F_valid, t2, MAX_LEN, BATCH, N, FEAT_SIZE, args.feat)
	if args.feat != None:
		y_pdct = model.predict([new_X,new_F], batch_size=BATCH)
		loss2t = model.test_on_batch([new_X,new_F], new_Y, sample_weight=None)
	else:
		y_pdct = model.predict(new_X, batch_size=BATCH)
		loss2t = model.test_on_batch(new_X, new_Y, sample_weight=None)
	loss2 = f_eval(y_pdct,new_Y)/ length
	total_loss2 += loss2
	total_loss2t+= loss2t

	ppl = np.e**(total_loss2/(t2+1))
	valid_ppl.append(ppl)


	sys.stdout.write( "\rEp:%i %i/%i train_loss:%f, train_acc:%f, val_loss:%f, ppl:%f, keras_ppl:%f...DONE!                                                    \
		\n" % (ep+1,t+1,DATA_LENGTH/BATCH,total_loss1/(t+1),total_acc1/(t+1),total_loss2/(t2+1),np.e**(total_loss2/(t2+1)),np.e**(total_loss2t/(t2+1))))
	#m_json = model.to_json()
	#open('model/LM_%s_%i_%i.json' % (args.feat, ep, FEAT_SIZE), 'w+').write(m_json)
	model.save_weights('weight/LM_%s_%i_%i.h5' % (args.feat, ep, FEAT_SIZE), overwrite=True)


index = valid_ppl.index(min(valid_ppl))

with open("results.txt", "a") as f:
	f.write("\rdata:%s, feat:%s, epoch:%s, ppl:%f" %(args.data, args.feat, index, valid_ppl[index]))

for i in range(EPOCH):
	if i != index:
		os.remove('weight/LM_%s_%s_%i.h5' % (args.feat, i, FEAT_SIZE))
	else:
		os.rename('weight/LM_%s_%s_%i.h5' % (args.feat, i, FEAT_SIZE), 'weight/LM_%s.h5' % (feat_name))


#TODO LIST:
#1. Train with word only
#2. Add features to training model
#3. Construct update platform
#4. TF-IDF for user-corpus
