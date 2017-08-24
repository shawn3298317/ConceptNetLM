from PosMap import *
from utils import *
from pattern.en import lemma
from collections import Counter
from sklearn.preprocessing import normalize
from itertools import chain

class POSFeature():
	def __init__(self):		
		self.Pos = readPosTag("nltk.txt")
		self.docs = np.load("../../corpus/background_200000.npy")
		#self.user = np.load("../../SocialNetwork/friend_corpus.npy").item()
		self.tag = dict()
		c = 0
		for t in self.Pos:
			self.tag[t] = c
			c += 1

	def pos_tagging(self):
		tags = []
		for sentence in self.docs:
			tag = list(nltk.pos_tag(sentence))
			tag.append(tag[0])
			tags.append(tag)
		return tags

	def createFeatVector(self, tags):
		self.tag = defaultdict(lambda: 40, self.tag)
		features = []
		for s in tags:
			feature = np.zeros((len(s), 41))
			for i in range(len(s)):
				index = self.tag[s[i][1]]
				feature[i][index] = 1.
			features.append(feature)
		features = np.asarray(features)
		return features

def main():
	test = POSFeature()
	tagged_sentences = test.pos_tagging()
	print "creating feature vector"
	feat = test.createFeatVector(tagged_sentences)
	np.save(open("../../data/feature/pos/naive_background.npy", "wb"), feat)
	
if __name__ == '__main__':
	main()

