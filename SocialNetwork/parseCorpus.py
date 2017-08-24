import json
import re
import nltk
import numpy as np

# Alternatives use 're' : its more convenient

def eliminate_syntax(word):
	
	word2 = word

	ret = (10, [word2,] )

	syntax = ['&gt;','&lt;','http://','RT','@','#','-']#,'...']
	quote = ['&quot;']#,'\\\\','\\',',',':','|','[',']','(',')']
	
		


	for syn in syntax:
		if syn in word2:
			'''
			if syn == "-":
				subs = word.split("-")
				if re.match(r'\A[\w-]+\Z', subs[0]) and re.match(r'\A[\w-]+\Z', subs[1]):
					if len(subs[0]) == 0:
						return (10, [subs[1:],])
					continue			
			'''	
			return (1,None)	


	quote_detect = False
	for q in quote:	
		if q in word2:
			word2 = word2.replace(q,'')
			#word.replace(' ','')
			#print 'replace %s' % word
			quote_detect = True
			#return (6, [word.replace(q,''),])
	if quote_detect == True:
		#word2.replace(' ','')
		if len(word2) == 0:
			return (1, None)
		return (6, [word2])
	return ret


def m_clarify(word):
	apos_cases = ['\'re','\'s','s\'','\'ll','\'ve','\'d',]
	ing_cases = ['in\'']
	w_cases = ['w/']

	toRet = []
	word = word.replace(")", "")
	word = word.replace("(", "")
	word = word.replace(" ", "")
	#Apostophe
	#tok = nltk.word_tokenize(word)
	#if len(tok) > 1:
	#	return (6, tok)
	
	for ap in apos_cases:
		if word.endswith(ap):
			toRet.append(word[:-1-len(ap)+1])
			if ap == 's\'':
				toRet.append('\'s')
			else:
				toRet.append(ap)
			#print 'APOS:',toRet
			return (6, toRet)
	

	#ING cases
	for ig in ing_cases:
		if word.endswith(ig):
			#print "Detect %s!" % ig
			toRet.append((word[:-1]+'g'))
			#print "Calib: ", toRet
			return (7, toRet)

	# w/ cases
	for case in w_cases:
		if word.startswith(case):
			toRet.append('with')
			toRet.append(word[2:])
			return (8, toRet)

	if word.isdigit():
		return (9, word)

	#Other cases replace non_alpha
	regex = re.compile('[^a-zA-Z-]')
	#print "word",word
	word = regex.sub("",word)
	#print word
	if len(word) > 0 and word != "":
		return (9, word)
	else:
		return (-1, None)

def Parse():

	Corpus = dict()
	with open('user_text.txt','r') as f:
		i = 0
		for line in f:
			user = line.split(" ")[0]
			if user not in Corpus:
				Corpus[user] = []
			#print i

			#if i > 200000:
			#	break
			#print line
			#print line
			sentence = line.split()[1:]
			sentence[1] = sentence[1][2:]
			#print sentence
			#raw_input()

			# Eliminate Syntax
			toAdd = []
			for word in sentence:
				valid, ret = eliminate_syntax(word)
				if valid > 5:
					#for w in ret:
					#toAdd.append(w.lower())
					rid , r = m_clarify(ret[0].lower())
					if rid == 9:
						#print r
						toAdd.append(r)
					elif rid > 0:
						for rr in r:
							#print rr
							toAdd.append(rr)
	
			#print toAdd
			#print ' '.join(toAdd)
			if len(toAdd) >= 1 and toAdd not in Corpus[user]:
				Corpus[user].append(toAdd)
			print toAdd
			raw_input()
	for user in Corpus:
		sentences_len = [len(x) for x in Corpus[user]]
		sentences_len, Corpus[user] = (list(t) for t in zip(*sorted(zip(sentences_len, Corpus[user]))))
		Corpus[user] = Corpus[user][-1000:]
		print "user", user, "contain sentences' len from", len(Corpus[user][0]), "to", len(Corpus[user][-1])
	#print Corpus["polascheps"]
	#np.save('background_corpus.npy',Corpus)
	#for user in Corpus:
	np.save(open("user_corpus_1000.npy", "wb"), Corpus)

Parse()
