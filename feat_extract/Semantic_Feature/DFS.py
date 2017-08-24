import operator



class DFS:

	def __init__(self, G):
		self.G = G
		self.path = []
		self.discovered = []
		self.err_attempt = 0
		self.depth = 8
		self.nodes = G.nodes()
		self.end = False
		self.found = False

	def find(self, src, target):
		self.reset()
		if src == target:
			return []

		if src not in self.nodes or target not in self.nodes:
			return []

		paths = self.DFS_search(src, target, [])
		return paths

	def reset(self):
		self.discovered = []
		self.depth = 5
		self.paths = []
		self.err_attempt = 0

	def DFS_search(self, src, target, path=[]):
		self.discovered.append(src)
		if len(self.paths) >= 3 or len(path) >= self.depth or self.err_attempt > 40000:
			return self.paths
		
		#if src in target:
		#	self.paths.append(path)

		neigh = self.G.neighbors(src)
		new_neigh = []
		for idx in range(len(neigh)):
			val = self.G.edge[src][neigh[idx]][0]['weight']
			new_neigh.append((neigh[idx], val))
		neigh, _ = zip(*sorted(new_neigh, key=operator.itemgetter(1), reverse=True))[:100]
		self.err_attempt += len(neigh)
		
		if target in neigh:
			path.append([self.G[src][target][0]['weight'], self.G[src][target][0]['rel']])
			self.paths.append(path)
		
		if len(self.paths) < 3 or len(path) < self.depth or self.err_attempt < 40000:			
			for idx in range(len(neigh)):
				if neigh[idx] in path or neigh[idx] in self.discovered: # resist lookback
					continue
				if len(neigh[idx].split()) > 1: # more than one word
					continue
				vertex = [self.G[src][neigh[idx]][0]['weight'], self.G[src][neigh[idx]][0]['rel']]
				self.DFS_search(neigh[idx], target, path+[vertex])
	
		#print self.err_attempt, len(path), len(self.paths)
		return self.paths
			
			
