# -*- coding: utf-8 -*-
##################################################################
#
# TUM Informatics
# Master Thesis Project: Indexing Methods for Social Search
# 
# Author: Oriana Baldizan
# Date:   March 2016
#
# stack_network.py : defines the social network of the Stack data
#
##################################################################

# Python Modules
import igraph
import matplotlib.pyplot as plt
from igraph import *
from numpy import histogram
from collections import Counter

# Project Modules
from stack_importer import StackImporter


class StackNetwork(object):
	""" Stackoverflow Social Network"""
	
	def __init__(self, setting):

		self.setting        = setting
		self.stack_importer = StackImporter(setting)
		self.stack_graph    = igraph.Graph()
		self.users          = self._load_users() # Stack users' ids


	def _load_users(self):
		""" Load the users from the specified db """

		self.stack_importer.open_stack_db()
		users = self.stack_importer.get_all_users()
		self.stack_importer.close_stack_db()

		return users

	def add_stack_users(self):
		""" Adds the list of active users to the network """

		for id in self.users:
			self.stack_graph.add_vertex(str(id))

	def add_users_connections(self):
		""" Add a list of edges (vertex1,vertex2) to the network """

		self.stack_importer.open_stack_db()

		for id in self.users:
			edges = self.stack_importer.get_user_friends(id)
			for e in edges:
				self.stack_graph.add_edge(str(e[0]), str(e[1]))

		self.stack_importer.close_stack_db()

		# Remove multiple edges between same pair of vertices
		self.stack_graph.simplify(multiple=False, loops=True, combine_edges='first')

	def print_network(self):

		print self.stack_graph.summary(verbosity=0)

	def calculate_statistics(self):
		""" Calculate and print important statistics to analyse the network """

		graph_components = self.stack_graph.components(mode=STRONG) # or graph.clusters(mode=STRONG)
		num_components   = len(graph_components.subgraphs())
		print "Number of strongly connected components = {}".format(num_components)

		biggest_component   = graph_components.giant()
		component_num_nodes = len(biggest_component.vs)
		component_num_edges = len(biggest_component.es)
		print "Nodes of the biggest strongly connected component = {}".format(component_num_nodes)
		print "Edges of the biggest strongly connected component = {}".format(component_num_edges)

		print "Graph clustering coefficient " + str(biggest_component.transitivity_undirected())
		print "Graph diameter " + str(biggest_component.diameter(directed=False, unconn=True, weights=None))
		print "Graph density " + str(biggest_component.density(loops=False))

		
		# Calculate the betweenness centrality for each node
		graph_betweenness = biggest_component.betweenness(directed=False, cutoff=None, weights=None)
		nodes_betweenness = [round(x) for x in graph_betweenness]
		nodes, bins   = histogram(nodes_betweenness, bins=15) #Counter(nodes_betweenness)

		print '\nBetweenness centrality distribution '
		#print nodes
		#print '\n'
		#print bins

		plt.bar(bins[:-1], nodes, width = 5)
		plt.xlim(min(bins), max(bins))

		plt.xlabel('Betweenness of nodes')
		plt.ylabel('Number of nodes')
		#plt.title('')
		plt.show() 

		print '\nDegree centrality '
		neighborhood_size = biggest_component.neighborhood_size()
		nodes, bins   = histogram(neighborhood_size, bins=20) #Counter(nodes_betweenness)
		#print nodes
		#print '\n'
		#print bins

		plt.bar(bins[:-1], nodes, width = 2)
		plt.xlim(min(bins), max(bins))
		plt.xlabel('Degree')
		plt.ylabel('Number of nodes')
		plt.show()   


		#print neighborhood_size
		avg_friends = sum(neighborhood_size) / float(len(neighborhood_size))
		print '\nAverage number of friends ' + str(avg_friends)
		print 'Max number of friends ' + str(max(neighborhood_size))


	def save_graph():
		""" Saves the graph in a file for later visualization """

		graph_components  = self.stack_graph.components(mode=STRONG) # or graph.clusters(mode=STRONG)
		biggest_component = graph_components.giant()
		nodes_betweenness = map(round, biggest_component.betweenness())
		biggest_component.vs["betweenness"] = nodes_betweenness
		biggest_component.save(theme + '_graph.txt', format='gml')




if __name__ == '__main__':

	theme = "beer"

	setting = {
	'theme': theme,

	'stack_dbpath': '../data/Stack/' + theme + '.db',
	'esa_dbpath': '../data/ESA/esa.db',

	'folderprefix': '../data/',
	'experiments_folder': '../data/Experiments/' + theme + '/pr_curve_experiment_',
	'resultfolder': '../results/',
	'resultdbfolder': '../results/model/similarities.db',
	}

	stack_network = StackNetwork(setting)

	# Initialize the network
	stack_network.add_stack_users()
	stack_network.add_users_connections()

	# Print resulting graph
	#stack_network.print_network()

	# Calculates different statistics
	#stack_network.calculate_statistics()