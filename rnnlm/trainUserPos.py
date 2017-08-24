from keras.models import Sequential
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
import numpy as np

import time
import sys

#MODEL PARAMATERS DEFINE
FEATURE_ADD = False

'M_PARAMATERS = [FEATURE_ADD,]'

#TRAINING PARAMTERS DEFINE
EPOCH = 50
BATCH = 20
DATA_LENGTH = 0#50000
MAX_LEN = 0
N = 44482

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

def expandFeat(MAX_LEN, pos_feat):
	FEAT_LEN = len(pos_feat[0][0])
	POS = np.zeros((len(pos_feat), MAX_LEN, FEAT_LEN))
	for i in range(len(pos_feat)):
		l = MAX_LEN - len(pos_feat[i]) + 1 # last word also expand to zero
		if l > 0:
			mask = np.zeros((l, FEAT_LEN))
			pos_feat[i] = np.concatenate((pos_feat[i][:-1], mask))
			POS[i] = pos_feat[i]
	return POS

def flush_print(t, loss, acc, total, time1, time2, ep):

	toWrite = "\rEp:%i %i/%i loss:%f, acc:%f, per_batch:%.3fs, load_time:%.3fs " % (ep+1,t,total,loss,acc,time1,time2)
	toWrite += calc((total-t) * (time2+time1))
	sys.stdout.write(toWrite)
	sys.stdout.flush()


def init_model(MAX_LEN, FEAT_SIZE):
		'''MODEL PARAMATERS'''
		
		WORD_SIZE = 1#2721127
		FEAT_SIZE  = 46
		WORD_EMBED = 300#300	

		'''MODEL STRUCTURE'''


		word = Sequential()
		word.add(Embedding(input_dim=N, input_length=MAX_LEN, output_dim=WORD_EMBED))  #, mask_zero=True


		if FEATURE_ADD:
			feature = Sequential()
			feature.add(Reshape(input_shape=(MAX_LEN,FEAT_SIZE), dims=(MAX_LEN,FEAT_SIZE)))
			model = Sequential()
			model.add(Merge([word,feature], mode='concat'))
			model.add(LSTM(output_dim = 1000, input_shape=(MAX_LEN, FEAT_SIZE+WORD_EMBED), activation='sigmoid',return_sequences=True)) 
			model.add(Dropout(0.3))
			model.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=FEAT_SIZE+WORD_EMBED, activation='softmax'))
			model.add(Dropout(0.3))

		else:
			word.add(LSTM(output_dim = WORD_EMBED, input_shape=(MAX_LEN, WORD_EMBED), activation='sigmoid',return_sequences=True))
			word.add(Dropout(0.3))
			word.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=WORD_EMBED, activation='softmax'))
			word.add(Dropout(0.3))
			
		model_final = model if FEATURE_ADD else word
		print "Loading model weight: %s..." % weights_path
		model_final.load_weights(weights_path)

		#sgd = SGD(lr = 0.01, decay = 1e-6, momemtum = 0.9, nesterov = True)

		print "Compiling"
		model_final.compile(loss='categorical_crossentropy', optimizer='adam')
		#model_final.compile(loss='mse', optimizer='rmsprop')#, class_mode='categorical')
		#model_final.compile(loss='mse', optimizer=sgd)
		return model_final





def main():
	users = ["polascheps", "nayrod", "mitchel_emily4e", "hannlawrence", "rpattzpicspam",\
			 "lalalovato", "rj_acosta", "iamnappy_901xxx", "marytorres12", "teampadalecki",\
			 "cbreezy_4ever", "slworking", "tweetypie54", "shoescom_sales", "mainlabswebsite", \
			 "demi_loveatox3", "seemasugandh", "kush_420_report", "mystyle", "swtgeorgiabrwn", \
			 "sallytheshizzle", "princessgwenie", "daddony58", "gmorataya", "celebfanspage", \
			 "missoxygenrina3", "sandrawelling", "milessellydemz", "aclaysuper", \
			 "youuarenotalone", "mjjalways", "_supernatural_", "sexmenickj", "officialas", \
			 "surinotes", "drubisunirun", "heather_wolf", "forensicmama", "protruckr", \
			 "milla_swe", "jaxsk", "crazybsbfan31"]
	
	user = "polascheps"
	print "Loading X Y data...."
	X = np.load('../SocialNetwork/user/XY/X_%s.npy' % user)
	Y = np.load('../SocialNetwork/user/XY/Y_%s.npy' % user)
	POS = np.load('../feat_extract/POS_Feature/features/users/%s_pos_feat.npy' % user)
	
	DATA_LENGTH = len(X)
	MAX_LEN     = len(X[0])
	
	print "Start training...?"
	raw_input()
	print "Initializing model"
	model = init_model(MAX_LEN, FEAT_SIZE)
	

	for ep in range(EPOCH):
		
		total_loss = 0.0
		total_acc = 0.0
		for t in range(DATA_LENGTH/BATCH):
	
			time_st = time.time()
			
			x   = X[BATCH*t:BATCH*(t+1)]
			y   = Y[BATCH*t:BATCH*(t+1)]
			pos = POS[BATCH*t:BATCH*(t+1)]
		
			timesteps = len(x[-1]) # sentences sorted in ascending order of length(0,...,30)
		
			new_X   = np.zeros((BATCH,timesteps)) # time series sentence
			new_Y   = np.zeros((BATCH,timesteps,44573))
			new_POS = np.zeros((BATCH,timesteps,46))
	
			for n in range(BATCH):
				for i in range(len(new_Y[n])):
					#print n,i
					#raw_input()
					idx = y[n][i]
					new_Y[n][i][idx] = 1.0
					new_X[n][i]      = x[n][i]
					new_POS[n][i]    = pos[n][i] 
	
			time_md2 = time.time()
	
			loss, acc =  model.train_on_batch([new_X, new_POS], new_Y, accuracy=True, class_weight=None, sample_weight=None)
			total_loss += loss
			total_acc += acc
	
			time_en = time.time()
			flush_print(t+1,total_loss/(t+1),total_acc/(t+1),DATA_LENGTH/BATCH, time_en-time_md2, time_md2-time_st, ep)
		sys.stdout.write( "\rEp:%i %i/%i loss:%f , acc:%f ...DONE!                                                     \
			\n" % (ep+1,t+1,DATA_LENGTH/BATCH,total_loss/(t+1),total_acc/(t+1)))
	
	print "Saving Model Data...."
	#raw_input()
	
	
	# m_json = model.to_json()
	# open('model/LM.json', 'w+').write(m_json)
	# model.save_weights('weight/LM.h5')
	# print "Save complete!"

if __name__ == '__main__':
	main()


#TODO LIST:
#1. Train with word only
#2. Add features to training model
#3. Construct update platform
#4. TF-IDF for user-corpus