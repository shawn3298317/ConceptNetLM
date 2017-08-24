from keras.models import Sequential
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
import numpy as np

import time
import sys
from utils import *
#MODEL PARAMATERS DEFINE
FEATURE_ADD = True

'M_PARAMATERS = [FEATURE_ADD,]'

#TRAINING PARAMTERS DEFINE
EPOCH = 50
BATCH = 10
DATA_LENGTH = 24000#10000
VAL_LENGTH  = 2400
MAX_LEN = 29
N = 48600#44573#103192


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

def flush_print(t, loss1, acc1, total, time1, time2, ep):

	toWrite = "\rEp:%i %i/%i loss:%f, acc:%f per_batch:%.3fs, load_time:%.3fs " % (ep+1,t,total,loss1,acc1,time1,time2)
	toWrite += calc((total-t) * (time2+time1))
	sys.stdout.write(toWrite)
	sys.stdout.flush()


def init_model():

		'''MODEL PARAMATERS'''
		
		WORD_SIZE = 1#2721127
		FEAT_SIZE  = 46
		WORD_EMBED = 300#300
		

		'''MODEL STRUCTURE'''


		word = Sequential()
		word.add(Embedding(input_dim=N, input_length=MAX_LEN, output_dim=WORD_EMBED))  #, mask_zero=True


		if FEATURE_ADD:
			feature = Sequential()
			#feature.add(Reshape(input_shape=(MAX_LEN,FEAT_SIZE), dims=(MAX_LEN,FEAT_SIZE)))
			feature.add(LSTM(output_dim = 46, input_shape=(MAX_LEN, FEAT_SIZE), activation='sigmoid',return_sequences=True))
			model = Sequential()
			model.add(Merge([word,feature], mode='concat'))
			model.add(LSTM(output_dim = 800, input_shape=(MAX_LEN, FEAT_SIZE+WORD_EMBED), activation='sigmoid',return_sequences=True)) 
			model.add(Dropout(0.3))
			model.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=800, activation='softmax'))
			model.add(Dropout(0.3))

		else:
			word.add(LSTM(output_dim = WORD_EMBED, input_shape=(MAX_LEN, WORD_EMBED), activation='sigmoid',return_sequences=True))
			word.add(Dropout(0.3))
			word.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=WORD_EMBED, activation='softmax'))
			word.add(Dropout(0.3))
			

		
		model_final = model if FEATURE_ADD else word

		#sgd = SGD(lr = 0.01, decay = 1e-6, momemtum = 0.9, nesterov = True)

		print "Compiling"
		model_final.compile(loss='categorical_crossentropy', optimizer='adam')
		#model_final.compile(loss='mse', optimizer='rmsprop')#, class_mode='categorical')
		#model_final.compile(loss='mse', optimizer=sgd)
		return model_final



model = init_model()

print "Loading X Y data...."

X_train = np.load('../SocialNetwork/X.npy')
Y_train = np.load('../SocialNetwork/Y.npy')
X_train, X_test, Y_train, Y_test = splitTrainTest(X_train, Y_train, VAL_LENGTH)

#print X.shapesplitTrainTest(corpus)
#print Y.shape

print "Start training...?"
raw_input()


for ep in range(EPOCH):

	total_loss1 = 0.0
	total_acc1 = 0.0
	total_loss2 = 0.0
	total_acc2 = 0.0
	
	for t in range(DATA_LENGTH/BATCH):

		time_st = time.time()
		
		x = X_train[BATCH*t:BATCH*(t+1)]
		y = Y_train[BATCH*t:BATCH*(t+1)]
		
		timesteps = len(x[-1]) # sentences sorted in ascending order of length(0,...,30)
		
		new_X = np.zeros((BATCH,timesteps)) # time series sentence
		new_Y = np.zeros((BATCH,timesteps,N))
		new_FEAT = np.zeros((BATCH,timesteps,46))

		for n in range(BATCH):
			for i in range(len(y[n])):
				#print n,i
				#raw_input()
				idx = y[n][i]
				new_Y[n][i][idx] = 1.0
				new_X[n][i] = x[n][i]

		time_md2 = time.time()

		#print 'X_batch shape:', new_X.shape
		#print 'X_batch shape:', new_Y.shape

		loss1, acc1 = model.train_on_batch([new_X, new_FEAT], new_Y, accuracy=True, class_weight=None, sample_weight=None)
		total_loss1 += loss1
		total_acc1 += acc1
		
		time_en = time.time()
		flush_print(t+1,total_loss1/(t+1),total_acc1/(t+1),DATA_LENGTH/BATCH, time_en-time_md2, time_md2-time_st, ep)

	for t2 in range(VAL_LENGTH/BATCH):
		x = X_test[BATCH*t2:BATCH*(t2+1)]
		y = Y_test[BATCH*t2:BATCH*(t2+1)]
		#feat = FEAT[BATCH*t:BATCH*(t+1)]
		timesteps = len(x[-1])
		new_X = np.zeros((BATCH,timesteps))
		new_Y = np.zeros((BATCH,timesteps,N))
		new_FEAT = np.zeros((BATCH,timesteps,46))
		for n in range(BATCH):
			for i in range(len(y[n])):
				idx = y[n][i]
				new_Y[n][i][idx] = 1.0
				new_X[n][i] = x[n][i]
		loss2, acc2 = model.test_on_batch([new_X,new_FEAT], new_Y, accuracy=True, sample_weight=None)
		total_loss2 += loss2
		total_acc2 += acc2


	sys.stdout.write( "\rEp:%i %i/%i train_loss:%f, train_acc:%f, val_loss:%f , val_acc:%f , ppl:%f...DONE!                                                    \
		\n" % (ep+1,t+1,DATA_LENGTH/BATCH,total_loss1/(t+1),total_acc1/(t+1),total_loss2/(t2+1),total_acc2/(t2+1),2**(total_loss2/(t2+1))))

	print "Saving Model Data...."
	m_json = model.to_json()
	open('model/LM_background_no_feat_%i.json' % ep, 'w+').write(m_json)
	model.save_weights('weight/LM_background_no_feat_%i.h5' % ep)

#raw_input()



# print "Save complete!"





#TODO LIST:
#1. Train with word only
#2. Add features to training model
#3. Construct update platform
#4. TF-IDF for user-corpus
