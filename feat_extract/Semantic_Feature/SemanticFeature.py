import sys
import nltk
import operator
import numpy as np
from pattern.en import lemma
from sklearn.preprocessing import normalize


class SemFeature():
	def __init__(self):
		self.sparse_dict = np.load("../../divisi/sparse_dict_all.npy").item()
		self.Rels = ["Synonym", "IsA", "RelatedTo", "HasA", "CreatedBy", "PartOf", "AtLocation", "LocatedNear", "DefinedAs",\
			"SymbolOf", "ReceivesAction", "HasPrerequisite", "MadeOf", "HasProperty", "UsedFor", "MotivatedByGoal",\
			"CapableOf", "Desires", "CausesDesire", "HasLastSubevent", "HasFirstSubevent", "HasSubevent", "Causes",\
			"NotIsA", "NotRelatedTo", "NotHasA", "NotCreatedBy", "NotPartOf", "NotAtLocation", "NotLocatedNear", \
			"NotDefinedAs", "NotSymbolOf", "NotReceivesAction", "NotHasPrerequisite", "NotMadeOf", "NotHasProperty",\
			"NotUsedFor", "NotMotivatedByGoal", "NotCapableOf", "NotDesires", "NotCausesDesire", "NotHasLastSubevent",\
			"NotHasFirstSubevent", "NotHasSubevent", "NotCauses", "Antonym", "DerivedFrom", "MemberOf"]
		
		self.docs = np.load("../../SocialNetwork/friend_corpus.npy").item()
		self.target_pair = ["n,n", "n,v", "n,adj", "n,adv", "v,v", "v,n", "v,adj", "v,adv"]

	def posDict(self):
		self.pos_dict = dict()
		with open("pos_dict.txt") as f:
			for line in f:
				pos = line.split(" ")[1] #NP
				self.pos_dict[pos] = line.split(" ")[0]

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
				#rel_seq.append(self.Tag["%s,%s" % (tag_seq[i-1], tag_seq[i])])
				pos_pair = "%s,%s" % (self.pos_dict[t[i  ][1]], self.pos_dict[t[i+1][1]])
				'''if pos_pair in 
				'''word_rel.append([t[i][0], self.Tag["%s,%s" % (curr_tag, next_tag)],\
								 self.pos_prob[t[i]]])
			parsed_sentences.append(word_rel)
		return parsed_sentences


	def createConceptPair(self, user):
		sentences_conceptPair = []
		for sentence in self.docs[user]:
			sentence_conceptPair = []
			sentence += [sentence[0]]
			for i in range(len(sentence) - 1):
				#if sentence[i+1] == "'s":
				#	next_word = lemma(sentence[i+2]).encode('utf-8')
				#else:
				next_word = lemma(sentence[i+1]).encode('utf-8')
				curr_word = lemma(sentence[i]).encode('utf-8')
				sentence_conceptPair.append([next_word, None, curr_word])
			sentences_conceptPair.append(sentence_conceptPair)
		return sentences_conceptPair

	def createFeatVector(self, sentences_conceptPair):
		num_of_sentences = len(sentences_conceptPair)
		features = []
		for i in range(num_of_sentences):
			sentence_len = len(sentences_conceptPair[i])
			feature = np.zeros((sentence_len, 48))
			sys.stdout.write("\rCreateing feature sentence %i / %i" % (i, num_of_sentences))
			sys.stdout.flush()
			for j in range(sentence_len):
				paths = self.DFS(sentences_conceptPair[i][j], sentences_conceptPair[i][j][0],\
					0, 8, feature[j], [])
				feature[j] = self.DFStoFeature(paths, feature[j])
				#self.BFS(sentences_conceptPair[i][j], 5, feature[j])
			feature = normalize(feature, axis=1)
			features.append(feature)
		return features

	def DFStoFeature(self, paths, feature):
		if len(paths) > 0:
			path_score = []
			for path in paths:
				score = 0.
				for node in path[1:]:
					if node[1] in self.Rels:
						score += node[0] / len(path[1:])
				path_score.append(score)
			path_score, paths = zip(*sorted(zip(path_score, paths),key=operator.itemgetter(0), reverse=True))
			for path in paths:
				for node in path[1:]:
					feature[self.Rels.index(node[1])] += node[0] / len(path[1:])
		return feature

	def DFS(self, start, end, depth, depth_l, feature, paths, path=None):
		result = []
		if start[2] not in self.sparse_dict or end not in self.sparse_dict:
			return result
		if path is None:
			path = [start]
		if end in start[2]:
			paths.append(path)
		#if end in zip(*self.sparse_dict[start[2]])[2]: #BFS with depth 1
		#	paths.append(path)
		if len(paths) == 3:
			return paths
		if depth < depth_l:
			children = [child for child in self.sparse_dict[start[2]] if child[0] > 0.1]
			children = sorted(children, key=operator.itemgetter(0), reverse=True)
			for vertex in children:
				depth += 1
				self.DFS(vertex, end, depth, depth_l, feature, paths, path+[vertex])
		if len(paths) > 3:
			return paths[:3]
		return paths

	#def BFS(self, pair, depth, feature):
	#	queue = [[pair]]
	#	path_count = 0
	#	if pair[0] not in self.sparse_dict or pair[2] not in self.sparse_dict:
	#		return feature
	#	while queue:
	#		tmp_path = queue.pop(0)
	#		if len(tmp_path) > depth:
	#			break
	#		last_node = tmp_path[-1]
	#		if pair[0] == last_node[2] and last_node[1] != None:
	#			path_count += 1
	#			for node in tmp_path[1:]:
	#				if node[1] in self.Rels:
	#					feature[self.Rels.index(node[1])] += node[0]
	#			if path_count > 5:
	#				#print "found 5 paths"
	#				break
	#		else:
	#			try:
	#				children = [child for child in self.sparse_dict[last_node[2]] if child[0] > 2.]
	#				children = sorted(children, key=operator.itemgetter(0), reverse=True)
	#				#print children
	#				for child in children:
	#					if child not in tmp_path:
	#						new_path = tmp_path + [child]
	#						queue += [new_path]
	#			except:
	#				pass
	#	print "%i path found" % path_count
	#	return feature

def main():
	a = SemFeature()
	user_counter = 0
	for user in a.docs:
		user_counter += 1
		print "\rUser_%i" % user_counter, user, "creating concept pair..."
		pair = a.createConceptPair(user)	
		print user, "has %i sentences" % len(pair)
		feat = a.createFeatVector(pair)
		#feats[user] = feat
		np.save(open("features/friend_3/%s_semantic_feat.npy" % user, "wb"), feat)

print "start..."
main()


