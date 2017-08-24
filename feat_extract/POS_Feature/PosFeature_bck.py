from PosMap import *
from utils import *
from pattern.en import lemma
from collections import Counter
from sklearn.preprocessing import normalize
from itertools import chain

class POSFeature():
	def __init__(self):		
		self.sparse_dict = np.load("../../divisi/sparse_dict_all.npy").item()
		self.Pos = readPosTag("nltk.txt")
		self.Tag, self.Rels = defRelation()
		self.docs = np.load("../../SocialNetwork/background_corpus.npy")
		#self.friends = np.load("../../SocialNetwork/friend_corpus.npy").item()
		
	def pos_tagging(self, sentences):
		tags = []
		for sentence in sentences:
			tag = nltk.pos_tag(sentence)
			tags.append(tag)
		return tags

	def parseTag2Rel(self, tagged_sentence):
		self.Pos = defaultdict(lambda: None, self.Pos)
		self.Tag = defaultdict(lambda: [], self.Tag)
		parsed_sentences = []
		for t in tagged_sentence:
			word_rel = []
			t.append(t[0])

			for i in range(len(t) - 1):
				curr_tag = self.Pos[t[i  ][1]]
				next_tag = self.Pos[t[i+1][1]]
				word_rel.append([t[i][0], self.Tag["%s,%s" % (curr_tag, next_tag)],\
								 self.pos_prob[t[i]]])
			parsed_sentences.append(word_rel)
		return parsed_sentences

	def createFeatVector(self, word_relations_list):
		features = []
		for sentence in word_relations_list:
			sentence_len = len(sentence)
			feature = np.zeros((sentence_len, 46))
			emissions = np.zeros((sentence_len, 1))
			if sentence_len == 0:
				continue
			for i in range(len(sentence)):
				word, rels, emission = sentence[i]
				#relation_weights = cn.search(lemma(word).encode('utf-8'))
				word = lemma(word).encode('utf-8')
				if word in self.sparse_dict:
					cn_weights, cn_rels = zip(*self.sparse_dict[word])[:2]
					for j in range(len(cn_rels)):
						if cn_rels[j] in rels:
							rel_index = self.Rels.index(cn_rels[j])
							feature[i][rel_index] += cn_weights[j]
					emissions[i] += emission
			# Normalization	
			feature = normalize(feature, axis=1) * emissions
			features.append(feature)
		features = np.asarray(features)
		return features

	def wordEmission(self, tagged_sentences):
		corpus = list(chain.from_iterable(tagged_sentences))
		count_result = Counter(corpus).most_common()
		word_dict = dict()
		self.pos_prob = {}
		for word in count_result:
			if word[0][0] not in word_dict:
				word_dict[word[0][0]] = [[word[0][1], word[1]]]
			else:
				word_dict[word[0][0]].append([word[0][1], word[1]])
		for word in word_dict:
			total = sum([freq for pos, freq in word_dict[word]])
			for pos, freq in word_dict[word]:
				self.pos_prob[(word, pos)] = float(freq) / total
		return self.pos_prob

def main():
	test = POSFeature()
	print "Background creating concept pair..."
	tagged_sentences = test.pos_tagging(test.docs)
	print "Calculating words' emission"
	test.wordEmission(tagged_sentences)
	print "parsing Tag to Relations"
	word_rel = test.parseTag2Rel(tagged_sentences)
	print "creating feature vector"
	feat = test.createFeatVector(word_rel)
	np.save(open("features/background_pos_feat.npy", "wb"), feat)
		
if __name__ == '__main__':
	main()

