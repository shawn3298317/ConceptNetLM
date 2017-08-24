import networkx as nx
import numpy as np
import time

def create_graph():
	a = nx.read_edgelist('../Json_rawData/assertions/Pairs/all.graph',\
		data=True, delimiter='\t',create_using=nx.MultiGraph())

	print "Total selfloops : ", a.number_of_selfloops()
	print "Removing selfloop edges..."
	a.remove_edges_from(a.selfloop_edges())
	a = a.to_undirected()

	# print "Save to gml..."
	# nx.write_gml(a, 'ConceptNett.xml')
	# print "Save complete!"
	print "Save to pkl..."
	nx.write_gpickle(a, "ConceptNet.pkl")
	return a

create_graph()