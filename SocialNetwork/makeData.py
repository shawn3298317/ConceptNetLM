import nltk
import numpy as np
from utils import *

def main():
	users = ['topofstuff', 'nieuwslogr', 'porngus', 'yyz1', 'site_news', 'seo_consultant', 'ocbarista', 'hotnews26', 'lingonews', 'flying_gramma', 'gluetext', 'ocnighter', 'topnews25', 'pflive_announce', 'annuitypayments', 'hot_bp', 'bestofyoutubers', 'delicious50', 'financetip', 'articlescreek', 'triplekillsblog', 'articles_mass', 'articlesmob', 'jobhits', 'mariolavandeira', 'sudhir_vashist', 'tubebutler', 'getfreelancejob', 'spitzezeug', 'rob_madden', 'car__tips', 'fooshare', 'blocalbargains', 'valuescompanies', 'techflypaper', 'articles4author', 'newsweb2x', 'real_advice_', 'mumbaitimes', 'newstop_us', 'dragtotop', 'tiptop_trends_1']
	c = 0
	total_X = np.asarray([])
	total_Y = np.asarray([])
	corpus = []
	for user in users:
		c+=1
		print c, user
		user_corpus = np.load("../corpus/user/%s/test.npy" % user)
		corpus = np.caoncatenate((corpus, user_corpus))
		
		X,Y = createInputXY("../corpus/user/%s/test.npy" % user,"N_dict_40402.npy")
		total_X = np.concatenate((total_X, X))
		total_Y = np.concatenate((total_Y, Y))
	
	#corpus = np.load("../corpus/background/background_corpus.npy")
	#X,Y = createInputXY("../corpus/background/background_corpus.npy", "N_dict_40402.npy")
	l = [len(x) for x in total_X]
	ind = np.asarray(l)
	ind = ind.argsort()
	total_X = np.asarray(total_X)[ind]
	total_Y = np.asarray(total_Y)[ind]
	corpus  = np.asarray(corpus)[ind]
	np.save("../data/user/X_users_test.npy", total_X)
	np.save("../data/user/Y_users_test.npy", total_Y)
	np.save("../corpus/users_test.npy", corpus)
	#X,Y=createInputXY(corpus, "N_dict_40402.npy")
	#np.save("X_users.npy", X)
	#np.save("Y_users.npy", Y)


if __name__ == '__main__':
	main()
