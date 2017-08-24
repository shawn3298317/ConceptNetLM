from keras.models import Sequential
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from sklearn.metrics import log_loss
import numpy as np
import theano
import theano.tensor as T

import time
import sys

'M_PARAMATERS = [FEATURE_ADD,]'

#TRAINING PARAMTERS DEFINE
EPOCH = 15
BATCH = 20
DATA_LENGTH = 25000
MAX_LEN = 29
N = 48600#827#44573#103192
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

def flush_print(t, loss, acc, total):

	toWrite = "\rEval %i/%i, loss:%f, acc:%f     " % (t,total,loss,acc)
	#toWrite += calc((total-t) * (time2+time1))
	sys.stdout.write(toWrite)
	sys.stdout.flush()

def evaluate_ppl():
	y_h  = T.matrix()
	y_t  = T.matrix()
	h = T.nnet.categorical_crossentropy(y_h, y_t)

	ret = theano.function([y_t, y_h], h)

	return ret

def init_model(weights_path):
		WORD_SIZE = 1#2721127
		WORD_EMBED = 300#300

		word = Sequential()
		word.add(Embedding(input_dim=N, input_length=MAX_LEN, output_dim=WORD_EMBED, mask_zero=True))

		feature = Sequential()
		feature.add(LSTM(output_dim = FEAT_SIZE, input_shape=(MAX_LEN, FEAT_SIZE), activation='sigmoid',return_sequences=True))
		
		model = Sequential()
		model.add(Merge([word,feature], mode='concat'))
		model.add(LSTM(output_dim = 600, input_shape=(MAX_LEN, FEAT_SIZE+WORD_EMBED), activation='sigmoid',return_sequences=True)) 
		model.add(Dropout(0.3))
		model.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=600, activation='softmax'))
		model.add(Dropout(0.3))
		
		print "Compiling"
		model.load_weights(weights_path)

		model.compile(loss='categorical_crossentropy', optimizer='adam')
		return model

X_train = np.load("../data/background/X_train.npy")
X_valid = np.load("../data/background/X_valid.npy")
Y_train = np.load("../data/background/Y_train.npy")
Y_valid = np.load("../data/background/Y_valid.npy")

model = init_model('weight/LM_None_6_48.h5')
#n_dict = np.load('../SocialNetwork/N_dict_48600.npy')[()]
#n_dict = dict(map(reversed, n_dict.items()))

total_loss = 0.0
total_acc = 0.0
for t in range(DATA_LENGTH/BATCH):

	x = X_train[BATCH*t:BATCH*(t+1)]
	y = Y_train[BATCH*t:BATCH*(t+1)]

	timesteps = len(x[-1]) # sentences sorted in ascending order of length(0,...,30)
	new_X = np.zeros((BATCH,timesteps)) # time series sentence
	new_Y = np.zeros((BATCH,timesteps,N))
	new_FEAT = np.zeros((BATCH,timesteps,FEAT_SIZE))

	for n in range(BATCH):
		for i in range(len(y[n])):
			idx = y[n][i]
			new_Y[n][i][idx] = 1.0
			new_X[n][i] = x[n][i]
		

	y_pdct = model.predict([new_X, new_FEAT], batch_size=BATCH)

	#loss, acc = model.test_on_batch(new_X, new_Y, accuracy=True, sample_weight=None)
	#print loss, acc
	#print new_Y[-1].index(1)
	y_pdct = np.clip(y_pdct, 1e-15, 1 - 1e-15)
	y_pdct /= y_pdct.sum(axis=1)[:, np.newaxis]
	loss = -(new_Y * np.log(y_pdct)).sum() / y_pdct.shape[0]
	print loss
	#print log_loss(new_Y, y_pdct)
	#y_idx = np.argmax(y_pdct, axis=2)
	# predict = []
	# for y in y_idx:
	# 	sentence = []
	# 	for idx in y:
	# 		sentence.append(n_dict[idx])
	# 	predict.append(sentence)

	# print predict
	# raw_input()

	#print y_idx
	#cmd = raw_input()

	
	#flush_print(t+1,total_loss/(t+1),total_acc/(t+1),DATA_LENGTH/BATCH)
	#sys.stdout.write( "\rEp:%i %i/%i loss:%f , acc:%f ...DONE!                                                     \
	#	\n" % (ep+1,t+1,DATA_LENGTH/BATCH,total_loss/(t+1),total_acc/(t+1)))

#flush_print(t+1,total_loss/(t+1),total_acc/(t+1),DATA_LENGTH/BATCH)
print "\nEvaluation complete!"



'''
for ep in range(EPOCH):

	total_loss = 0.0
	total_acc = 0.0
	for t in range(DATA_LENGTH/BATCH):

		time_st = time.time()
		
		x = X[BATCH*t:BATCH*(t+1)]
		y = Y[BATCH*t:BATCH*(t+1)]

		new_Y = np.zeros((BATCH,32,44573))

		for n in range(BATCH):
			for i in range(32):
				idx = y[n][i]
				new_Y[n][i][idx] = 1.0

		time_md2 = time.time()

		loss, acc =  model.train_on_batch(x, new_Y, accuracy=True, class_weight=None, sample_weight=None)
		total_loss += loss
		total_acc += acc

		time_en = time.time()
		flush_print(t+1,total_loss/(t+1),total_acc/(t+1),DATA_LENGTH/BATCH, time_en-time_md2, time_md2-time_st, ep)
	sys.stdout.write( "\rEp:%i %i/%i loss:%f , acc:%f ...DONE!                                                     \
		\n" % (ep+1,t+1,DATA_LENGTH/BATCH,total_loss/(t+1),total_acc/(t+1)))

print "Saving Model Data...."
#raw_input()
'''

# m_json = model.to_json()
# open('model/LM.json', 'w+').write(m_json)
# model.save_weights('weight/LM.h5')

# print "Save complete!"





#TODO LIST:
#1. Train with word only
#2. Add features to training model
#3. Construct update platform
#4. TF-IDF for user-corpus
