import sys
import time
import nltk
import operator
import numpy as np
import networkx as nx
from pattern.en import lemma
from sklearn.preprocessing import normalize
from collections import defaultdict
from collections import Counter
import math
from DFS import *

local_count = {}
word_count = {}
loc_dict = {}

def get_tfidf(word):
	tf = float(local_count[word])/word_count[word]
	idf = math.log10(42.0/(1 + len(loc_dict.get(word, 0))))
	tfidf = tf*idf

	#print word, U_dict.get(word,-1),1+tfidf

	return tfidf

class SemFeature():
	def __init__(self):
		self.posDict()
		self.sparse_dict = np.load("../../divisi/sparse_dict_all.npy").item()
		#self.G = nx.read_gpickle('../../SocialNetwork/ConceptNet.pkl') # create_graph()
		#self.dfs = DFS(self.G)

		self.Rels = ["Synonym", "IsA", "RelatedTo", "HasA", "CreatedBy", "PartOf", "AtLocation", "LocatedNear", "DefinedAs",\
			"SymbolOf", "ReceivesAction", "HasPrerequisite", "MadeOf", "HasProperty", "UsedFor", "MotivatedByGoal",\
			"CapableOf", "Desires", "CausesDesire", "HasLastSubevent", "HasFirstSubevent", "HasSubevent", "Causes",\
			"NotIsA", "NotRelatedTo", "NotHasA", "NotCreatedBy", "NotPartOf", "NotAtLocation", "NotLocatedNear", \
			"NotDefinedAs", "NotSymbolOf", "NotReceivesAction", "NotHasPrerequisite", "NotMadeOf", "NotHasProperty",\
			"NotUsedFor", "NotMotivatedByGoal", "NotCapableOf", "NotDesires", "NotCausesDesire", "NotHasLastSubevent",\
			"NotHasFirstSubevent", "NotHasSubevent", "NotCauses", "Antonym", "DerivedFrom", "MemberOf"]
		
		self.docs = np.load("../../SocialNetwork/user_corpus_1000.npy").item()
		self.target = ["n", "v", "adj"]
		
	def posDict(self):
		self.pos_dict = dict()
		with open("pos_dict.txt") as f:
			for line in f:
				pos = line.split(" ")[1][:-1] #NP
				self.pos_dict[pos] = line.split(" ")[0]
		
	def posTagging(self, corpus):
		tags = []
		for sentence in corpus:
			tag = nltk.pos_tag(sentence)
			tags.append(tag)
		return tags

	def parseTag2Rel(self, tagged_corpus):
		self.pos_dict = defaultdict(lambda: None, self.pos_dict)
		target_corpus = []
		for t in tagged_corpus:
			target_words = []
			for i in range(len(t)):
				if self.pos_dict[t[i][1]] in self.target:
					target_words.append(t[i][0])
			#print target_words
			target_corpus.append(target_words)		
		return target_corpus

	def createConceptPair(self, corpus, target_corpus):
		corpus_conceptPair = []
		for sentence, target_sentence in zip(corpus,target_corpus):
			#print sentence
			#raw_input()
			sentence_conceptPair = []
			if len(target_sentence) > 0:
				target_sentence += [target_sentence[0]]
			for i in range(len(sentence)):
				if sentence[i] in target_sentence:
					curr_word = lemma(sentence[i]).encode('utf-8')
					next_word = lemma(target_sentence[target_sentence.index(sentence[i])+1]).encode('utf-8')
				else:
					curr_word = lemma(sentence[i]).encode('utf-8')
					next_word = curr_word
				#next_word = lemma(sentence[i+1]).encode('utf-8')
				#curr_word = lemma(sentence[i]).encode('utf-8')
				sentence_conceptPair.append([curr_word, next_word])
				#sentence_conceptPair.append([next_word, None, curr_word])
			corpus_conceptPair.append(sentence_conceptPair)
		return corpus_conceptPair
	def flush_print(self,i,total):

		toWrite = "\rUpdated %i/%i .." % (i, total)
		sys.stdout.write(toWrite)
		sys.stdout.flush()


	def createFeatVector(self, sentences_conceptPair):
		num_of_sentences = len(sentences_conceptPair)
		features = []
		for i in range(num_of_sentences):
			sentence_len = len(sentences_conceptPair[i])
			feature = np.zeros((sentence_len, 48))
			sys.stdout.write("\rCreateing feature sentence %i / %i" % (i, num_of_sentences))
			sys.stdout.flush()
			for j in range(sentence_len):
				self.paths = []
				self.err_attempt = 0
				#paths = self.dfs.find(sentences_conceptPair[i][j][0], sentences_conceptPair[i][j][1])		
				paths = self.DFS(sentences_conceptPair[i][j], sentences_conceptPair[i][j][0],\
					0, 6)
				feature[j] = self.DFStoFeature(paths, feature[j])
				#self.BFS(sentences_conceptPair[i][j], 5, feature[j])
			feature = normalize(feature, axis=1)
			features.append(feature)
		return features

	def DFStoFeature(self, paths, feature):
		if len(paths) > 0:
			for path in paths:
				decay = 1.
				for node in path[1:]:
					feature[self.Rels.index(node[1])] += node[0] / len(path[1:]) * decay
					decay *= 0.85
			#print feature
		return feature

	def DFS(self, start, end, depth, depth_l, path=[]):
		if len(self.paths) >= 3 or depth == depth_l or self.err_attempt >= 80000:
			#print self.err_attempt, len(self.paths), depth
			return self.paths
		if start[2] not in self.sparse_dict or end not in self.sparse_dict or start[2] == end:
			return self.paths
		
		path = [start] if [] else path
		if end in start[2]:
			self.paths.append(path)
		
		neigh = zip(*self.sparse_dict[start[2]])[2] #BFS with depth 1
		if end in neigh:
			idx = neigh.index(end)
			path += [self.sparse_dict[start[2]][idx]]
			self.paths.append(path)
		
		self.err_attempt += 1
		if depth < depth_l:
			children = [child for child in self.sparse_dict[start[2]] if child[0] > 0.1]
			children = sorted(children, key=operator.itemgetter(0), reverse=True)
			for vertex in children:
				depth += 1
				self.DFS(vertex, end, depth, depth_l, path+[vertex])
		return self.paths

def main():
	a = SemFeature()
	user_counter = 0
	
	word_count = np.load('wordcount.npy')[()]
	loc_dict = np.load('loc_dict.npy')[()]
	#print "Loading graph..."
        #time_0 = time.time()
        #G = nx.read_gpickle('cn.pkl') # create_graph()
	#dfs = DFS(G)
        #print "Complete in %fs" % (time.time()- time_0)	


	#print len(a.docs)	

	for user in a.docs:
		
		print "Loading graph..."
	        time_0 = time.time()
        	G = nx.read_gpickle('cn.pkl') # create_graph()
	        dfs = DFS(G)
        	print "Complete in %fs" % (time.time()- time_0)		

		print "Creating %s tfidf dict..." % user
		corpus = a.docs[user]
		tfidf = {}

		text = []
		[ text.extend(el) for el in corpus]
		local_count = Counter(text)
		
		for word in local_count.keys():
			tf = float(local_count[word])/word_count[word]
		        idf = math.log10(42.0/(1 + len(loc_dict.get(word, 0))))
		        score = tf*idf
			tfidf[word] = (1 + score)
		
		#print tfidf
		
		user_counter += 1
                print "\rUser_%i" % user_counter, user, "creating concept pair..."
                tagged_corpus = a.posTagging(a.docs[user])
                target_corpus = a.parseTag2Rel(tagged_corpus)
                pair = a.createConceptPair(a.docs[user], target_corpus)
		
		num = 0.0
		
		for ps in pair:
                        #t0 = time.time()
			num += 1
			a.flush_print(num,len(pair))
			#print num,"/",tot
                        for p in ps:
                                t0 = time.time()
                                #print "find (%s %s)" % (p[0], p[1])
                               	path =  dfs.find(p[0], p[1])
                                #print "Found :" , path
				#print "Search time: %fs" % (time.time()-t0)
				
				if len(path) > 0:
					new_weight = tfidf.get(path[0], 1.0)				
				else:
					continue

				neigh = G.neighbors(p[0])

				# Diffuse
				for n in neigh:
					G[p[0]][n][0]['weight'] *= new_weight				
					

				# Infiltrate
				for i in range(len(path)):
					if i == len(path)-1:
						break

					#print "%s new_w : %f" % (path[i], new_weight)
					G[path[i]][path[i+1]][0]['weight'] *= new_weight
					#raw_input()
		
		print "Save %s graph to pkl..." % user
		nx.write_gpickle(G, "Graph/mix/update_%s.graph"%user)
		print "Save complete!"		


print "start..."
main()


