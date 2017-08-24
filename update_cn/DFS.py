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

		self.DFS_search(src, target)

		if self.found:
			#print "Found : ", self.path
			self.path.insert(0,src)
			self.path.append(target)
			return self.path
		else:
			#print "Cannot find..."
			return self.path


	def reset(self):

		self.path = []
		self.discovered = []
		self.err_attempt = 0
		self.depth = 8
		self.end = False
		self.found = False




	def DFS_search(self, src, target):

		neigh = self.G.neighbors(src)
		self.discovered.append(src)
		#print "(%s %s) %i" % (src,target,self.depth)
		#print self.path
		#raw_input()

		if target in neigh:
			self.found = True
			#print 'found in level', 8-self.depth, self.G[src][target][0]#, self.path
			self.end = True
			return

		elif self.depth == 0:
			if self.err_attempt < 10000:
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
			for idx in range(len(neigh)):
				if neigh[idx] in self.path or neigh[idx] in self.discovered:
					continue
				if len(neigh[idx].split()) > 1:
					continue
				#val = self.G.edge[src][neigh[idx]][0]['weight']
				self.path.append(neigh[idx])
				
				#print "->(%s %s) %i" % (neigh[idx],target,self.depth)
				self.DFS_search(neigh[idx], target)
				if self.end == True:
					return
				#neigh_dict[neigh[idx]] = val
				#new_neigh.append((neigh[idx], val))
				# if val > max_val:
				# 	max_val = val
				# 	max_idx = idx

			#sorted_neigh = sorted(new_neigh, key=operator.itemgetter(1))
			#print sorted_neigh[:10] , neigh[:10]

			# for n in sorted_neigh:
			# 	self.path.append(n[0])
			# 	self.depth -= 1
			# 	self.DFS_search(n[0], target)

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
			self.path.pop()
			
			return

			#return



			# if max_idx == -1:
			# 	#print "Backing off!"
			# 	if len(self.path) == 1:
			# 		print "Traversed whole graph, cannot find path..."
			# 	self.path = self.path[:-1]
			# 	self.depth += 1
			# 	self.DFS_search(self.path[-1],target)
			# 	return

			# #print '(%s %s)' % (s, neigh[max_idx])
			# else:
			# 	self.path.append(neigh[max_idx])
			# 	self.depth -= 1
			# 	self.DFS_search(neigh[max_idx], target)
			# return 

	#def sort(neigh)

