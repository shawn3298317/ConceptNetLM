from DFS import *
import networkx as nx


G = nx.read_gpickle('../../update_cn/cn.pkl') # create_graph
dfs = DFS(G)
print "start finding"
path = dfs.find("thirst", "hot")

for p in path:
	print len(p), p