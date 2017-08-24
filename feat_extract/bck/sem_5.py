import sys
import nltk
import operator
import numpy as np
import networkx as nx
from pattern.en import lemma
from sklearn.preprocessing import normalize
from collections import defaultdict
import pdb
from DFS2 import *

''' 
two line needs to be changed,
line 33 & 166
seperate it into 8000-15000 sentences a thread
'''

class SemFeature():
	def __init__(self):
		self.posDict()
		self.user = "algebra_com"
		self.G = nx.read_gpickle('../../update_cn/cn.pkl')
		#self.G = nx.read_gpickle('../../update_CN/Graph_updated/path/%s_update_p.pkl' % self.user)
		self.dfs = DFS(self.G,6,3)
		#self.sparse_dict = np.load("../../divisi/sparse_dict_all.npy").item()
		
		self.Rels = ["Synonym", "IsA", "RelatedTo", "HasA", "CreatedBy", "PartOf", "AtLocation", "LocatedNear", "DefinedAs",\
			"SymbolOf", "ReceivesAction", "HasPrerequisite", "MadeOf", "HasProperty", "UsedFor", "MotivatedByGoal",\
			"CapableOf", "Desires", "CausesDesire", "HasLastSubevent", "HasFirstSubevent", "HasSubevent", "Causes",\
			"NotIsA", "NotRelatedTo", "NotHasA", "NotCreatedBy", "NotPartOf", "NotAtLocation", "NotLocatedNear", \
			"NotDefinedAs", "NotSymbolOf", "NotReceivesAction", "NotHasPrerequisite", "NotMadeOf", "NotHasProperty",\
			"NotUsedFor", "NotMotivatedByGoal", "NotCapableOf", "NotDesires", "NotCausesDesire", "NotHasLastSubevent",\
			"NotHasFirstSubevent", "NotHasSubevent", "NotCauses", "Antonym", "DerivedFrom", "MemberOf"]
		
		self.docs = np.load("../../corpus/user/valid.npy")#np.load("../../corpus/background_200000.npy")[:100000]
		self.target = ["n", "v", "adj"]
		self.depth_l = 6

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

	def createConceptPair(self):
		sentences_conceptPair = []
		for sentence in self.docs:
			sentence_conceptPair = []
			sentence = list(sentence)
			sentence += [sentence[0]]
			for i in range(len(sentence) - 1):
				prev_word = lemma(sentence[i-1]).encode('utf-8')
				next_word = lemma(sentence[i+1]).encode('utf-8')
				curr_word = lemma(sentence[i]).encode('utf-8')
				sentence_conceptPair.append([ [curr_word, next_word], [curr_word, prev_word] ])
				#sentence_conceptPair.append([next_word, None, curr_word])
			sentences_conceptPair.append(sentence_conceptPair)
		return sentences_conceptPair

	def createFeatVector(self, sentences_conceptPair):
		#wrf = open('features/sem_feat.txt', 'a')
		#with open('features/sem_feat.txt', 'a') as wrf:
		num_of_sentences = len(sentences_conceptPair)
		features = []
		for i in range(num_of_sentences):
			#wrf.write('%s\n' % (i+1))
			sentence_len = len(sentences_conceptPair[i])
			feature = np.zeros((sentence_len, 96))
			sys.stdout.write("\rCreateing feature sentence %i / %i" % (i, num_of_sentences))
			sys.stdout.flush()
			for j in range(sentence_len):
				paths = self.dfs.find(sentences_conceptPair[i][j][0][0], sentences_conceptPair[i][j][0][1])		
				#paths = self.DFS(sentences_conceptPair[i][j][0], sentences_conceptPair[i][j][0][0],0)
				feature[j][:48] = self.DFStoFeature(paths, feature[j][:48])
				
				paths = self.dfs.find(sentences_conceptPair[i][j][1][0], sentences_conceptPair[i][j][1][1])		
				#paths = self.DFS(sentences_conceptPair[i][j][1], sentences_conceptPair[i][j][1][0],0)
				feature[j][48:] = self.DFStoFeature(paths, feature[j][48:])
				#self.BFS(sentences_conceptPair[i][j], 5, feature[j])
			feature = normalize(feature, axis=1)
			#np.savetxt(wrf, feature, delimiter=' ', newline='\n')
			features.append(feature)
		#wrf.close()
		return features

	def DFStoFeature(self, paths, feature):
		if len(paths) > 0:
			for path in paths:
				if len(path) > 0:
					decay = 1.
					for node in path:
						feature[self.Rels.index(node[1])] += node[0] / len(path) * decay
						decay *= 0.8
			#print feature
		return feature

	def DFS(self, start, end, depth, path=[]):
		#print len(self.paths), depth, self.err_attempt
		if len(self.paths) >= 3 or depth >= self.depth_l or self.err_attempt >= 50000:
			return self.paths
		if start[2] not in self.sparse_dict or end not in self.sparse_dict or start[2] == end:
			return self.paths
		#path = [start] if [] else path
		if end in start[2]:
			self.paths.append(path)
			return self.paths
		
		neigh = zip(*self.sparse_dict[start[2]])[2] #BFS with depth 1
		self.err_attempt += len(neigh)
		if end in neigh:
			idx = neigh.index(end)
			path += [self.sparse_dict[start[2]][idx]]
			self.paths.append(path)
			return self.paths
		
		self.err_attempt += 1
		if depth < self.depth_l:
			children = [child for child in self.sparse_dict[start[2]] if child[0] > 0.1]
			children = sorted(children, key=operator.itemgetter(0), reverse=True)
			for vertex in children:
				self.DFS(vertex, end, depth+1, path+[vertex])
		return self.paths

def main():
	a = SemFeature()
	#print "pos tagging"
	#tagged_corpus = a.posTagging(a.docs)
	#print "parsing tag 2 relation"
	#target_corpus = a.parseTag2Rel(tagged_corpus)
	print "creating concept pair"
	pair = a.createConceptPair()
	###		
	feat = a.createFeatVector(pair)
	#np.save(open("../../data/feature/sem/user/%s_background_1.npy" % self.user, "wb"), feat)
	np.save(open("../../data/feature/sem/user_valid_naive6.npy.npy", "wb"), feat)	

print "start..."
main()


