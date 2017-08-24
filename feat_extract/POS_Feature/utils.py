import sys
import numpy as np
from collections import defaultdict
from conceptnet5_client.web import api
from conceptnet5_client.web.api import LookUp, Search


class ConceptNet():
	def __init__(self):		
		self.lookup = api.LookUp(limit=1000)

	def search(self, concept):
		self.response = self.lookup.search_concept(concept)
		assert(len(self.response) == 2)	
		relation_weights = dict()
		relation_weights = defaultdict(lambda: [], relation_weights)
		for index, result in enumerate(self.response["edges"]):
			for line in result.items()[2][1]:
				if "/r/" in line:
					line = line.split("/r/")[1]
					if "/" not in line:
						relation = line[:-2]
			relation_weights[relation].append(result.items()[-8][1])

		for item in relation_weights:
			relation_weights[item] = sum(relation_weights[item]) / len(relation_weights[item])
		return relation_weights
