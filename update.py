import networkx as nx
import numpy as np
import time
from DFS import *

# def DFS_search(G, s, target, path, err, depth = 8):

# 	# if depth == -1:
		

# 	# 	print "not found!"
# 	# 	return None

# 	neigh = G.neighbors(s)
# 	#raw_input()

# 	if target in neigh:
# 		print 'found in level', 8-depth, G[s][target]
# 		return path
# 	elif depth == 0:
# 		n_err = 

# 	else:
# 		max_idx = 0
# 		max_val = 0.0
# 		for idx in range(len(neigh)):
# 			if neigh[idx] in path:
# 				continue
# 			val = G.edge[s][neigh[idx]][0]['weight']
# 			if val > max_val:
# 				max_val = val
# 				max_idx = idx
# 		#print '(%s %s)' % (s, neigh[max_idx])
# 		path.append(neigh[max_idx])
# 		DFS_search(G, neigh[max_idx], target, path, depth-1)



print "Loading graph..."
time_0 = time.time()
G = nx.read_gpickle('ConceptNet.pkl') # create_graph()
#G = nx.read_gml('ConceptNett.xml')
print "Complete! ", time.time()- time_0


print "Load corpus..."
#corpus = np.load('../../../divisi/update_cn/corpus/user/jaxsk_corpus.npy')
corpus = np.load("user/1000/articles_mass.npy")

#time_st = time.time()

dfs = DFS(G)
#node_list = G.nodes()

#dfs.find('show','glee')

counter = 0 
for sentence in corpus:
	#raw_input()
	counter += 1
	print counter 
	for i in range(len(sentence)):
		#raw_input()
		time_st = time.time()
		if i == len(sentence)-2: 
			break
		print "Finding (",sentence[i], sentence[i+1], ")"
		#dfs.DFS_search(sentence[i], sentence[i+1], 8)
		dfs.find(sentence[i], sentence[i+1])

		print "Search time: %f\n" % (time.time()- time_st) 
		#raw_input()

#print "Time: ", time.time()-time_st
#n = G.nodes()
#n = G.edge['pahoehoe']['aa']
#neigh = G.neighbors('anything')

#p = DFS_search(G, 'eat', 'shit', ['anything',], 8)

#print neigh
'''
for n in neigh:
	print "%s\t%s : " % ("anything",n) , G.edge['anything'][n][0]['weight']
	#raw_input()
	G.edge['anything'][n][0]['weight'] *= 1.1





print "Modified..."
for n in neigh:
	print "%s\t%s : " % ("anything",n) , G.edge['anything'][n][0]['weight']
'''

# anything , shit

