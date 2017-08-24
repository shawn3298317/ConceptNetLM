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
		#self.user = "algebra_com"
		self.G = nx.read_gpickle('../../update_cn/cn.pkl')
		self.dep = 3
		self.n_path = 3
		#self.G = nx.read_gpickle('../../update_CN/Graph_updated/path/%s_update_p.pkl' % self.user)
		self.dfs = DFS(self.G, self.dep, self.n_path)
		#self.sparse_dict = np.load("../../divisi/sparse_dict_all.npy").item()
		
		self.Rels = ["Synonym", "IsA", "RelatedTo", "HasA", "CreatedBy", "PartOf", "AtLocation", "LocatedNear", "DefinedAs",\
			"SymbolOf", "ReceivesAction", "HasPrerequisite", "MadeOf", "HasProperty", "UsedFor", "MotivatedByGoal",\
			"CapableOf", "Desires", "CausesDesire", "HasLastSubevent", "HasFirstSubevent", "HasSubevent", "Causes",\
			"NotIsA", "NotRelatedTo", "NotHasA", "NotCreatedBy", "NotPartOf", "NotAtLocation", "NotLocatedNear", \
			"NotDefinedAs", "NotSymbolOf", "NotReceivesAction", "NotHasPrerequisite", "NotMadeOf", "NotHasProperty",\
			"NotUsedFor", "NotMotivatedByGoal", "NotCapableOf", "NotDesires", "NotCausesDesire", "NotHasLastSubevent",\
			"NotHasFirstSubevent", "NotHasSubevent", "NotCauses", "Antonym", "DerivedFrom", "MemberOf"]
		
		self.docs = dict()
		self.docs["background_2_naive"] = np.load("../../corpus/background_200000.npy")[100000:]
		#self.docs["train_naive"] = np.load("../../corpus/user/train.npy")
		self.docs["valid_naive"] = np.load("../../corpus/user/valid.npy")
		self.docs["test_naive"]  = np.load("../../corpus/user/test.npy")
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

	def createConceptPair(self, filename):
		sentences_conceptPair = []
		for sentence in self.docs[filename]:
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
		return feature

	def Semantic(self, filename):
		#print "pos tagging"
		#tagged_corpus = self.posTagging(self.docs[filename])
		#print "parsing tag 2 relation"
		#target_corpus = self.parseTag2Rel(tagged_corpus)
		print "creating concept pair"
		pair = self.createConceptPair(filename)
		feat = self.createFeatVector(pair)
		np.save(open("../../data/feature/sem/%s_d%i_p%i_2.npy" % (filename, self.dep, self.n_path), "wb"), feat)	


def main():
	a = SemFeature()
	a.Semantic("background_2_naive")
	#a.Semantic("train_naive")
	a.Semantic("valid_naive")
	a.Semantic("test_naive")

print "start..."
main()


