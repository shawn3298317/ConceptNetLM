import numpy as np
from spacy.en import English

docs = dict()
docs["background"] = np.load("../../corpus/background_200000.npy")
docs["train"] = np.load("../../corpus/user/train.npy")
docs["valid"] = np.load("../../corpus/user/valid.npy")
docs["test"]  = np.load("../../corpus/user/test.npy")

nlp = English()

feat = dict()
feat["background"] = []
feat["train"] = []
feat["valid"] = []
feat["test"]  = []
for c in docs:
	for s in docs[c]:
		f = []
		for w in s:
			f.append(nlp(u"%s" % w).vector)
		feat[c].append(f)

for c in feat:
	np.save("../../data/feature/w2v/%s.npy" % c, feat[c])
			
