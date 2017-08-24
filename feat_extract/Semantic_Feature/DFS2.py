import operator



class DFS:

	def __init__(self, G):
		self.G = G
		self.path = []
		self.discovered = {}
		self.err_attempt = 0
		self.depth = 8
		self.nodes = G.nodes()
		self.end = False
		self.found = False
		self.node_dict = {}

		for n in self.nodes:
			self.node_dict[n] = 1.0

	def find(self, src, target):

		self.reset()

		
		if src == target:
			#print "Same word!"
			return []

		if src not in self.node_dict or target not in self.node_dict:
			#print "Not in ConceptNet!"
			return []
		
		for i in range(3):
			self.DFS_search(src, target)
			
			if self.found:
				#print "Found : ", self.path
				self.path.insert(0,src)
				self.path.append(target)
				#return self.path
			#else:
				#print "Cannot find..."
				#return self.path
			self.paths.append(self.path)

			self.path = []
			self.depth = 5
			self.end = False
			self.found = False

		#return self.paths
		relations = []
		for p in self.paths:
			relation = []
			for i in range(len(p)):
				if i == len(p)-1:
					continue
				pair = (self.G.edge[p[i]][p[i+1]][0]['weight'],self.G.edge[p[i]][p[i+1]][0]['rel'])
				relation.append(pair)
			relations.append(relation)
		return relations
			
		

	def reset(self):

		self.path = []
		self.paths = []
		self.discovered = {}
		self.err_attempt = 0
		self.depth_l = 6
		self.depth = 5
		self.end = False
		self.found = False


	def DFS_search(self, src, target):

		neigh = self.G.neighbors(src)
		#self.discovered.append(src)
		self.discovered[src] = True

		if target in neigh:
			self.found = True
			#print 'found in level', 8-self.depth, self.G[src][target][0]#, self.path
			self.end = True
			return

		elif self.depth == 0:
			if self.err_attempt < 5000:
				self.err_attempt += 1
				self.path.pop()
				#print "Discovering new_path"
				#self.DFS_search(self.path[-1],target, depth+1)
				#print "returning..."
				#self.depth += 1
				return
			
			else:
				#print "Cannot find valid path!", self.err_attempt
				self.end = True
				return

		else:
			#print "len:",len(neigh)
			self.depth -= 1
			sort_neigh = []
			for idx in range(len(neigh)):
				if neigh[idx] in self.path or neigh[idx] in self.discovered:
					continue
				if len(neigh[idx].split()) > 1:
					continue
				
				val = self.G.edge[src][neigh[idx]][0]['weight']
				#self.path.append(neigh[idx])
				#print "->(%s %s) %i" % (neigh[idx],target,self.depth)
				#self.DFS_search(neigh[idx], target)
				#if self.end == True:
				#	return
				
				
				sort_neigh.append((neigh[idx], val))
				

			sort_neigh = sorted(sort_neigh, key=operator.itemgetter(1), reverse=True)
			#print sorted_neigh[:10] , neigh[:10]

			for n in sort_neigh:
				self.path.append(n[0])
				#self.depth -= 1
				self.DFS_search(n[0], target)
				if self.end == True:
					return

			# if len(self.path) == 1:
			# 	print "Traversed whole graph, cannot find path..."
			# 	return
			#if len(neigh) == 0:
			#	print "No neighbor!"
			#else:
			if len(self.path) == 0:
				#print "Traversed whole graph, cannot find path..."
			 	return
			self.depth += 1
			if len(self.path) != 0:
				self.path.pop()
			
			return

	def DFS_search_hit(self, src, target):

		neigh = self.G.neighbors(src)
		#self.discovered.append(src)
		self.discovered[src] = True

		if target in neigh:
			self.found = True
			#print 'found in level', 8-self.depth, self.G[src][target][0]#, self.path
			self.end = True
			return

		elif self.depth == 0:
			if self.err_attempt < 5000:
				self.err_attempt += 1
				self.path.pop()
				#print "Discovering new_path"
				#self.DFS_search(self.path[-1],target, depth+1)
				#print "returning..."
				#self.depth += 1
				return
			
			else:
				#print "Cannot find valid path!", self.err_attempt
				self.end = True
				return

		else:
			#print "len:",len(neigh)
			self.depth -= 1
			sort_neigh = []
			for idx in range(len(neigh)):
				if neigh[idx] in self.path or neigh[idx] in self.discovered:
					continue
				if len(neigh[idx].split()) > 1:
					continue
				
				val = self.G.edge[src][neigh[idx]][0]['weight']
				#self.path.append(neigh[idx])
				#print "->(%s %s) %i" % (neigh[idx],target,self.depth)
				#self.DFS_search(neigh[idx], target)
				#if self.end == True:
				#	return
				
				
				sort_neigh.append((neigh[idx], val))
				

			sort_neigh = sorted(sort_neigh, key=operator.itemgetter(1), reverse=True)
			#print sorted_neigh[:10] , neigh[:10]

			for n in sort_neigh:
				self.path.append(n[0])
				#self.depth -= 1
				self.DFS_search(n[0], target)
				if self.end == True:
					return

			# if len(self.path) == 1:
			# 	print "Traversed whole graph, cannot find path..."
			# 	return
			#if len(neigh) == 0:
			#	print "No neighbor!"
			#else:
			if len(self.path) == 0:
				#print "Traversed whole graph, cannot find path..."
			 	return
			self.depth += 1
			if len(self.path) != 0:
				self.path.pop()
			
			return


