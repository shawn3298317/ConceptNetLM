from PosMap import *
from utils import *
from pattern.en import lemma

class POSFeature():
	def __init__(self):		
		self.Pos = readPosTag("nltk.txt")
		self.Tag, self.Rels = defRelation()
		self.docs = ["Today's weather is good.", "This is what I want to eat."]
		self.lookup = api.LookUp(limit=1000)

	def pos_tagging(self, sentences):
		tags = []
		for sentence in sentences:
			print sentence
			tagged_s = []
			tokens = nltk.word_tokenize(sentence)
			tags.append(nltk.pos_tag(tokens))
		return tags

	def parseTag2Rel(self, tagged_sentence):
		self.Pos = defaultdict(lambda: None, self.Pos)
		word_rel = []
		for t in tagged_sentence:
			tag_seq = []
			rel_seq = []
			doc = copy.copy(t)
			for word in doc:
				if self.Pos[word[1]] != None:
					tag_seq.append(self.Pos[word[1]])
				else:
					t.remove(word)
			tag_seq.append(tag_seq[0])
			for i in range(len(t)):
				rel_seq.append(self.Tag["%s,%s" % (tag_seq[i-1], tag_seq[i])])
				word_rel.append([t[i][0], self.Tag["%s,%s" % (tag_seq[i-1], tag_seq[i])]])		
		return word_rel

	def createFeatVector(self, word_relations_list):
		sentence_len = len(word_relations_list)
		feature = np.zeros((sentence_len, 46))
		for i in range(len(word_relations_list)):
			word, rels = word_relations_list[i]
			relation_weights = cn.search(lemma(word).encode('utf-8'))
			for j in range(len(self.Rels)):
				if self.Rels[j] in relation_weights and self.Rels[j] in rels:
					feature[i][j] = relation_weights[self.Rels[j]]
		return feature

cn = ConceptNet()
test = POSFeature()
tagged_sentences = test.pos_tagging(test.docs)
word_rel = test.parseTag2Rel(tagged_sentences)
v = test.createFeatVector(word_rel)


