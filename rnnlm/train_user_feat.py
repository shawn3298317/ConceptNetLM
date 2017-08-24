from keras.models import Sequential
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
import numpy as np
import argparse
import os
import time
import sys
from utils import *

#MODEL PARAMATERS DEFINE
FEATURE_ADD = True

'M_PARAMATERS = [FEATURE_ADD,]'

#TRAINING PARAMTERS DEFINE
EPOCH = 15
BATCH = 12
DATA_LENGTH = 480#10000
VAL_LENGTH  = 160
TEST_LENGTH = 160
MAX_LEN = 29
N = 48600#44573#103192
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

def flush_print(t, loss, acc, total, time1, time2, ep):

	toWrite = "\rEp:%i %i/%i loss:%f, acc:%f, per_batch:%.3fs, load_time:%.3fs " % (ep+1,t,total,loss,acc,time1,time2)
	toWrite += calc((total-t) * (time2+time1))
	sys.stdout.write(toWrite)
	sys.stdout.flush()


def init_model(weights_path):

		'''MODEL PARAMATERS'''
		
		WORD_SIZE = 1#2721127
		WORD_EMBED = 300#300
		'''MODEL STRUCTURE'''


		word = Sequential()
		word.add(Embedding(input_dim=N, input_length=MAX_LEN, output_dim=WORD_EMBED, mask_zero=True))


		if FEATURE_ADD:
			feature = Sequential()
			feature.add(LSTM(output_dim = FEAT_SIZE, input_shape=(MAX_LEN, FEAT_SIZE), activation='sigmoid',return_sequences=True))
			model = Sequential()
			model.add(Merge([word,feature], mode='concat'))
			model.add(LSTM(output_dim = 600, input_shape=(MAX_LEN, FEAT_SIZE+WORD_EMBED), activation='sigmoid',return_sequences=True)) 
			model.add(Dropout(0.3))
			model.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=600, activation='softmax'))
			model.add(Dropout(0.3))

		else:
			word.add(LSTM(output_dim = WORD_EMBED, input_shape=(MAX_LEN, WORD_EMBED), activation='sigmoid',return_sequences=True))
			word.add(Dropout(0.3))
			word.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=WORD_EMBED, activation='softmax'))
			word.add(Dropout(0.3))
			

		
		model_final = model if FEATURE_ADD else word

		print "Loading model weight: %s..." % weights_path
		model_final.load_weights(weights_path)

		print "Compiling"
		model_final.compile(loss='categorical_crossentropy', optimizer='adam')
		#model_final.compile(loss='mse', optimizer='rmsprop')#, class_mode='categorical')
		#model_final.compile(loss='mse', optimizer=sgd)
		return model_final

parser = argparse.ArgumentParser()
parser.add_argument('-feat', type=str)
parser.add_argument('-user', type=int)
args = parser.parse_args()

users = userList("../SocialNetwork/list.txt")
user  = users[args.user]
feat  = args.feat

print "-----------------------\nUser: %s \nFeature: %s" % (user, feat)
	
print "-----------------------\nLoading X Y data...."

if feat != None:
	X_train, X_valid, X_test, Y_train, Y_valid, Y_test, F_train, F_valid, F_test = \
					loadUserData(user, feat)
else:
	X_train, X_valid, X_test, Y_train, Y_valid, Y_test = \
					loadUserData(user, feat)

feat_size = 48 if feat == "sem" else 46

print "Initializing..."
model = init_model('weight/saved/LM_%s.h5' % feat)

print "Start training..."
#raw_input()
valid_ppl = []
for ep in range(EPOCH):
	total_loss1 = 0.0
	total_acc1 = 0.0
	total_loss2 = 0.0
	total_acc2 = 0.0
	#TRAIN
	for t in range(DATA_LENGTH/BATCH):

		time_st = time.time()
		
		x    = X_train[BATCH*t:BATCH*(t+1)]
		y    = Y_train[BATCH*t:BATCH*(t+1)]
		if feat != None:
			feat = F_train[BATCH*t:BATCH*(t+1)]
		timesteps = len(x[-1]) # sentences sorted in ascending order of length(0,...,30)
		new_X    = np.zeros((BATCH,timesteps)) # time series sentence
		new_Y    = np.zeros((BATCH,timesteps,N))
		new_FEAT = np.zeros((BATCH,timesteps,FEAT_SIZE))

		for n in range(BATCH):
			for i in range(len(y[n])):
				idx = y[n][i]
				new_Y[n][i][idx] = 1.0
				new_X[n][i] = x[n][i]
				if args.feat != None:
					new_FEAT[n][i][:feat_size] = np.asarray(feat[n][i][:feat_size])

		time_md2 = time.time()
		loss1, acc1 =  model.train_on_batch([new_X, new_FEAT], new_Y, accuracy=True, class_weight=None, sample_weight=None)
		
		total_loss1 += loss1
		total_acc1 += acc1
		
		time_en = time.time()
		flush_print(t+1,total_loss1/(t+1),total_acc1/(t+1),DATA_LENGTH/BATCH, time_en-time_md2, time_md2-time_st, ep)

	#VALID
	for t2 in range(VAL_LENGTH/BATCH):
		x = X_valid[BATCH*t2:BATCH*(t2+1)]
		y = Y_valid[BATCH*t2:BATCH*(t2+1)]
		if feat != None:
			feat = F_valid[BATCH*t2:BATCH*(t2+1)]
		
		timesteps = len(x[-1])
		
		new_X = np.zeros((BATCH,timesteps))
		new_Y = np.zeros((BATCH,timesteps,N))
		new_FEAT = np.zeros((BATCH,timesteps,FEAT_SIZE))

		for n in range(BATCH):
			for i in range(len(y[n])):
				idx = y[n][i]
				new_Y[n][i][idx] = 1.0
				new_X[n][i] = x[n][i]
				if args.feat != None:
					new_FEAT[n][i][:feat_size] = np.asarray(feat[n][i][:feat_size])

		loss2, acc2 = model.test_on_batch([new_X,new_FEAT], new_Y, accuracy=True, sample_weight=None)
		total_loss2 += loss2
		total_acc2 += acc2
	ppl = 2**(total_loss2/(t2+1))
	valid_ppl.append(ppl)
	sys.stdout.write( "\rEp:%i train_loss:%f, train_acc:%f, val_loss:%f , val_acc:%f , ppl:%f...DONE!\n" % \
		(ep+1,total_loss1/(t+1),total_acc1/(t+1),total_loss2/(t2+1),total_acc2/(t2+1),2**(total_loss2/(t2+1))))
	
	m_json = model.to_json()
	open('model/LM_%s_%s_%i.json' % (user, args.feat, ep), 'w+').write(m_json)
	model.save_weights('weight/LM_%s_%s_%i.h5' % (user, args.feat, ep), overwrite=True)

# TEST
index = valid_ppl.index(min(valid_ppl))
model = init_model('weight/LM_%s_%s_%i.h5' % (user, args.feat, index))
total_loss3 = 0.
total_acc3  = 0.
for t3 in range(TEST_LENGTH/BATCH):
	x = X_test[BATCH*t3:BATCH*(t3+1)]
	y = Y_test[BATCH*t3:BATCH*(t3+1)]
	if feat != None:
		feat = F_test[BATCH*t3:BATCH*(t3+1)]
	
	timesteps = len(x[-1])
	
	new_X = np.zeros((BATCH,timesteps))
	new_Y = np.zeros((BATCH,timesteps,N))
	new_FEAT = np.zeros((BATCH,timesteps,FEAT_SIZE))
	
	for n in range(BATCH):
		for i in range(len(y[n])):
			idx = y[n][i]
			new_Y[n][i][idx] = 1.0
			new_X[n][i] = x[n][i]
			if args.feat != None:
				new_FEAT[n][i][:feat_size] = np.asarray(feat[n][i][:feat_size])
	
	loss2, acc2 = model.test_on_batch([new_X,new_FEAT], new_Y, accuracy=True, sample_weight=None)
	total_loss3 += loss2
	total_acc3  += acc2

sys.stdout.write("\rUser:%s, feature:%s, Test_loss:%f, test_acc:%f, test_ppl:%f, train_ppl:%f, val_ppl:%f                                               \
	\n" % (user,args.feat,total_loss3/(t3+1),total_acc3/(t3+1),2**(total_loss3/(t3+1)),2**(total_loss1/(t+1)),2**(total_loss2/(t2+1))))

for i in range(15):
	if i != index:
		os.remove('weight/LM_%s_%s_%i.h5' % (user, args.feat, i))
	else:
		os.rename('weight/LM_%s_%s_%i.h5' % (user, args.feat, index), 'weight/LM_%s_%s.h5' % (user, args.feat))

with open("results.txt", "a") as f:
	f.write("\rUser:%s, feature:%s, Test_loss:%f, test_acc:%f, test_ppl:%f, train_ppl:%f                                                 \
		\n" % (user,args.feat,total_loss3/(t3+1),total_acc3/(t3+1),2**(total_loss3/(t3+1)),2**(total_loss1/(t+1))))


#TODO LIST:
#1. Train with word only
#2. Add features to training model
#3. Construct update platform
#4. TF-IDF for user-corpus
	