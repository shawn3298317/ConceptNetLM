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
FEATURE_ADD = True

'M_PARAMATERS = [FEATURE_ADD,]'

#TRAINING PARAMTERS DEFINE
EPOCH = 20
BATCH = 10
DATA_LENGTH = 142#50000


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
		l = MAX_LEN - len(pos_feat[i])
		if l > 0:
			mask = np.zeros((l, FEAT_LEN))
			pos_feat[i] = np.concatenate((pos_feat[i], mask))
			POS[i] = pos_feat[i]
	return POS

def flush_print(t, loss, acc, total, time1, time2, ep):

	toWrite = "\rEp:%i %i/%i loss:%f, acc:%f, per_batch:%.3fs, load_time:%.3fs " % (ep+1,t,total,loss,acc,time1,time2)
	toWrite += calc((total-t) * (time2+time1))
	sys.stdout.write(toWrite)
	sys.stdout.flush()


def init_model(MAX_LEN, FEAT_SIZE):

		'''MODEL PARAMATERS'''
		N = 44573#103192
		WORD_SIZE = 1#2721127
		WORD_EMBED = 300#1000


		'''MODEL STRUCTURE'''


		word = Sequential()
		word.add(Embedding(input_dim=N, input_length=MAX_LEN, output_dim=WORD_EMBED))  #, mask_zero=True


		if FEATURE_ADD:
			feature = Sequential()
			#feature.add(Reshape(input_shape=(MAX_LEN,FEAT_SIZE), dims=(MAX_LEN,FEAT_SIZE)))
			feature.add(LSTM(output_dim = 48, input_shape=(10, 48), activation='tanh', return_sequences=True))
			feature.add(add(Merge([in_cept_m, in_ques_m], mode='dot', dot_axes=([2],[1])))
			feature.add(TimeDistributedDense(output_dim=300, input_length=MAX_LEN, input_dim=300))

			model = Sequential()
			model.add(Merge([word,feature], mode='concat'))
			model.add(LSTM(output_dim = 1000, input_shape=(MAX_LEN, 300+WORD_EMBED), activation='tanh',return_sequences=True)) 
			model.add(Dropout(0.3))
			model.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=FEAT_SIZE+WORD_EMBED, activation='softmax'))
			model.add(Dropout(0.3))

		else:
			word.add(LSTM(output_dim = WORD_EMBED, input_shape=(MAX_LEN, WORD_EMBED), activation='tanh',return_sequences=True))
			word.add(Dropout(0.3))
			word.add(TimeDistributedDense(output_dim = N, input_length=MAX_LEN, input_dim=WORD_EMBED, activation='softmax'))
			word.add(Dropout(0.3))
		

		
		model_final = model if FEATURE_ADD else word

		print "Compiling"
		model_final.compile(loss='categorical_crossentropy', optimizer="adam")
		#model_final.compile(loss='mse', optimizer='rmsprop')#, class_mode='categorical')
		print "Compile finished"
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
	#POS = np.load('../feat_extract/POS_Feature/features/users/%s_pos_feat.npy' % user)
	SEM = np.load('../feat_extract/Semantic_Feature/features/user_time/%s_semantic_feat.npy' % user)
	print X.shape
	print SEM.shape
	#print Y.shape
	MAX_LEN  = len(X[-1])
	FEAT_SIZE = len(SEM[0][0][0])
	#POS = expandFeat(MAX_LEN, POS)
	#SEM = expandFeat(MAX_LEN, SEM)
	print "Start training...?"
	raw_input()
	print "Initializing model"
	model = init_model(MAX_LEN, FEAT_SIZE)
	for ep in range(EPOCH):
		total_loss = 0.0
		total_acc = 0.0
		for t in range(DATA_LENGTH/BATCH):
	
			time_st = time.time()
			
			x = X[BATCH*t:BATCH*(t+1)]
			y = Y[BATCH*t:BATCH*(t+1)]
			sem = SEM[BATCH*t:BATCH*(t+1)]

			timesteps = len(x[-1]) # sentences sorted in ascending order of length(0,...,30)
			f_timesteps = max([max([len(f) for f in s]) for s in sem])
			
			new_X = np.zeros((BATCH,timesteps)) # time series sentence
			new_Y = np.zeros((BATCH,timesteps,44573))
			new_SEM=np.zeros((BATCH,timesteps,f_timesteps,48))
			for n in range(BATCH):
				for i in range(len(x[n])):
					idx = y[n][i]
					new_Y[n][i][idx] = 1.0
					new_X[n][i] = x[n][i]
					for j in range(len(sem[n][i])):
						new_SEM[n][i][j] = sem[n][i][j]
			time_md2 = time.time()
	
			loss, acc =  model.train_on_batch([new_X, new_SEM], new_Y, accuracy=True, class_weight=None, sample_weight=None)
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