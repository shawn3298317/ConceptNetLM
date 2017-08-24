import json
import csv
import numpy as np
import nltk
import sys
# make user_text.txt
def targetUser():
	target_users = []
	with open('user_size_42.txt', 'r') as f:
		for line in f:
			target_users.append(line.split(' ')[0])
	print target_users
	return target_users

def userfriendCorpus():
	target_users = targetUser()
	uf = np.load("user_friend.npy").item()
	uf_c = dict()
	with open('Tweets.json') as f:
		for line in f:
			if '"iso_language_code": "en"' in line:
				user_name = line.split('"from_user": ')[1].split('"')[1]
				for u in uf:
					if user_name in uf[u]:
						text = line.split('"text": "')[1].split('", "from_user_id"')[0]
						if u not in uf_c:
							uf_c[u] = [text]
						else:
							uf_c[u].append(text)
	np.save("user_friend_corpus.npy", uf_c)

def makeFriendCorpus():
	uf_c = np.load("friend_corpus_4.npy").item()
	tu = targetUser()
	f_c = dict()
	for u in tu:
		if len(uf_c[u]) > 3000:
			print u
			i = 0
			N = np.load("user_N/N_%s.npy" % u).item()
			for s in uf_c[u]:
				i += 1
				sys.stdout.write("\r%i / %i" % (i, len(uf_c[u])))
				sys.stdout.flush()
				count = 0
				for w in s:
					if w in N:
						count += 1
				if count >= len(s)-1:
					if u not in f_c:
						f_c[u] = [s]
					else:
						f_c[u].append(s)
			#print "\n", len(f_c[u])
			for s in uf_c[u]:
				if len(f_c[u]) < 3000:
					if s not in f_c[u]:
						f_c[u].append(s)
		else:
			f_c[u] = uf_c[u][:3000]
		#f_c[u] = f_c[u][:2400]
		print "\n", len(f_c[u])
		
	np.save("friend_corpus_all.npy", f_c)

makeFriendCorpus()
#users_text = dict()
#users_n = dict()
#with open('Tweets.json') as f:
#	for line in f:
#		if '"iso_language_code": "en"' in line: 
#			user_name = line.split('"from_user": ')[1].split('"')[1]
#			if user_name in target_users:
#				text = line.split('"text": "')[1].split('", "from_user_id"')[0]
#				if user_name not in users_text:
#					users_text[user_name] = [text]
#				else:
#					users_text[user_name].append(text)
#			#if user_name not in users_n and "new" not in user_name:
#			#	users_n[user_name] = 0
#			#elif "new" not in user_name:
#			#	users_n[user_name]+= 1
#
#'''make users_text.txt'''
#print len(users_text), "users"
#with open('user_text.txt', 'wb') as f:
#	for u in users_text:
#		counter = 0
#		for sentence in users_text[u]:
#			if len(nltk.word_tokenize(sentence)) > 10:
#				counter += 1
#				f.write("%s %s\n" % (u, sentence))
#		print "user", u, "has", counter, "sentences"

#print len(users_n), " users"
#ls = []
#user_names = []
#for name in users_n:
#	user_names.append(name)
#	ls.append(users_n[name])
#
#user_indexes = sorted(range(len(ls)), key=lambda i: ls[i], reverse=True)
#ls, user_names = (list(t) for t in zip(*sorted(zip(ls, user_names))))



#make user_friend.txt
#user_friend = dict()
#with open('../UserGraph/USER_GRAPH.csv') as f:
#	datareader = csv.reader(f)
#	for row in datareader:
#		if row[0] in target_users:
#			if row[0] not in user_friend:
#				user_friend[row[0]] = [row[1]]
#			else:
#				user_friend[row[0]].append(row[1])
#		if row[1] in target_users:
#			if row[1] not in user_friend:
#				user_friend[row[1]] = [row[0]]
#			else:
#				user_friend[row[1]].append(row[0])
#
#uf_n = dict()
#for u in user_friend:
#	for f in user_friend[u]:
#		if f in users_n:
#			if u not in uf_n:
#				uf_n[u] = users_n[f]
#			else:
#				uf_n[u]+= users_n[f]



#with open("user_friend.txt", "wb") as f:
#	for user in user_friend:
#		for friend in user_friend[user]:
#			f.write("%s %s\n" % (user, friend))
#np.save("user_friend.npy", user_friend)
#

#user_indexes = sorted(range(len(ls)), key=lambda i: ls[i], reverse=True)[:100]
#f = open("user_size_friend.txt", 'wb')
#for u, n in zip(target_ucsers, ls):
#	if n > 1000 and u in uf_n and uf_n[u] > 1000:
#		f.write("%s %s %i\n" % (u, n, uf_n[u]))
#np.save('user_size.npy', user_indexes)

