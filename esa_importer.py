# -*- coding: utf-8 -*-
###########################################################################
#
# TUM Informatics
# Master Thesis Project: Indexing Methods for Social Search
# 
# Author: Oriana Baldizan
# Date:   October 2015
#
# esa_importer.py: Defines the methods that manage ESA data with SQLite
#
# - Creates a database for the ESA data
# - Manipulates and retrieves the data
#
###########################################################################

# Python Modules
from multiprocessing import Process, Pool
from collections import defaultdict, Counter
from operator import itemgetter
from gensim import corpora

import math
import sqlite3
import os.path
import logging
import struct  # To binarize the tf-idf vectors
import unicodedata

# Project Modules
from wiki_importer import WikiImporter, WikiCorpus



class EsaIndex(object):

	def __init__(self, connection):

		self.connection = connection


	def __iter__(self):

		self.connection.text_factory = str

		# Open database connection
		cursor = self.connection.cursor()
		query  = 'SELECT term, term_vector FROM term tv LEFT JOIN term_map tm ON tv.term_id = tm.term_id'
		cursor.execute(query)

		for term, term_vector in cursor:
			related_documents = []
			# Unpack binary string to vector form (8 comes from 'if' format)
			for i in xrange(0, len(term_vector), 8):
				document_id, value   = struct.unpack('if', term_vector[i:i+8])
				related_documents.append( (document_id,value) )

			new_documents = self.sliding_window_prunning(related_documents)

			yield (term, ESAImporter.pack_vectors(new_documents))


	def sliding_window_prunning(self, vectors, window_size=50, threshold=0.05):
		""" Applies the algorithm that eliminate spurious associations
		between articles (concepts) and words. See paper for reference. """

		# Sort all concepts by the TF-IDF value in decreasing order
		# @vectors is a list of tuples, the second element is the value
		sorted_list = sorted(vectors, key=itemgetter(1))
		sorted_list = sorted_list[::-1] # Decreasing order

		# Scan the sequence with a @window_size of 100, and truncate the
		# sequence when the difference in scores between the first and last 
		# concepts in the window drops below 5% of the highest-score
		max_score = sorted_list[0][1]
		result    = []
		append    = result.append

		for i, (document_id, value) in enumerate(sorted_list):

			if len(result) >= window_size:
				# First and last elements of the window
				window_first = sorted_list[ max(0, i - window_size) ]
				window_last  = sorted_list[ max(0, i - 1) ]

				# Subtract elements' values
				scores_dif = window_first[1] - window_last[1]

				# Check prunning condition
				if scores_dif <= max_score * threshold:
					break

			append( (document_id, value) )

		if len(result) < len(sorted_list):
			print "PRUNED SOMETHING " + str(len(result)) + "  " + str(len(sorted_list))
		else:
			print "NO PRUNING"

		return result



class ESAImporter (object):
	""" Manager for data used by the ESA algorithm """

	def __init__(self, setting):

		self.setting      = setting
		self.connection   = None
		self.testing      = self.setting['mini_test']

	def open_esa_db(self):

		if self.testing:
			self.connection = sqlite3.connect(self.setting['mini_esa_dbpath'])
		else:
			self.connection = sqlite3.connect(self.setting['esa_dbpath'])

	def close_esa_db(self):

		# Close connection
		self.connection.close()


	###############################################################################
	# Database initialization
	###############################################################################

	def create_esa_db(self):
		""" Initializes the ESA database """

		# Open database connection
		cursor = self.connection.cursor()

		#"DROP TABLE IF EXISTS doc_term_freq",
		drop_tables = [
		"DROP TABLE IF EXISTS term",
		"DROP TABLE IF EXISTS term_map"
		]

		#"CREATE TABLE doc_term_freq (term_id int, doc_id int, freq int)",
		create_tables = [
		"CREATE TABLE term (term_id INTEGER PRIMARY KEY, term_vector text)",
		"CREATE TABLE term_map (term text PRIMARY KEY, term_id int)"
		]

		for query in drop_tables:
			cursor.execute(query)

		for query in create_tables:
			cursor.execute(query)


	###############################################################################
	# Data saving and retrieval
	###############################################################################

	def get_wiki_corpus(self):
		""" Returns the Wikipedia's corpus """

		corpus = []

		# Get corpus from database
		result = WikiCorpus(sqlite3.connect(self.setting['wiki_dbpath']))

		for document in result:
			corpus.append( document.content.split(' ') )

		return corpus


	def get_wiki_corpus_dictionary(self):
		""" Returns the Wikipedia's dictionary and corpus """

		dictionary = {}

		# Get corpus from database
		corpus = WikiCorpus(self.connection)

		# Read dictionary
		cursor = self.connection.cursor()
		query  = 'SELECT term, term_id FROM term_map'
		cursor.execute(query)

		for term, term_id in cursor:
			dictionary[term] = term_id

		return (corpus, dictionary)


	def update_articles_content(self):

		dictionary = {}

		cursor = self.connection.cursor()
		query  = 'SELECT term FROM term_map'
		cursor.execute(query)

		for term in cursor:
			dictionary[term[0]] = ""

		connection = sqlite3.connect(self.setting['wiki_dbpath'])
		corpus     = WikiCorpus(connection, self.testing)

		query = 'CREATE TABLE wiki_articles (id INTEGER PRIMARY KEY, content text)'
		cursor.execute(query)
		self.connection.commit()

		logging.info("Updating corpus ...")
		values     = []
		append     = values.append
		
		for document in corpus:

			content = document.content.split(' ')

			new_content = [x for x in content if x in dictionary]
			print "Doc " + str(document.document_id)
			append( (document.document_id, ' '.join(new_content) ) )

		query = 'INSERT INTO wiki_articles VALUES (?, ?)'
		cursor.executemany(query, values)
		self.connection.commit()
		

		connection.close()



	def save_term_doc_frequencies(self):
		""" For each wiki article, calculates the frequency of each word in it """

		corpus          = WikiCorpus(self.connection)

		# Clean table
		cursor2 = self.connection.cursor()
		query   = "DROP TABLE IF EXISTS doc_term_freq"
		cursor2.execute(query)

		query   = "CREATE TABLE doc_term_freq (term_id int, doc_id int, freq int)"
		cursor2.execute(query)

		logging.info("Saving the term/document frequencies... ")
		cursor  = self.connection.cursor()
		query   = 'SELECT term, term_id FROM term_map'
		cursor.execute(query)

		# Keep dictionary
		dictionary = {}
		for word, word_id in cursor:
			dictionary[unicode(word)] = word_id

		for document in corpus:
			frequencies = []
			content     = document.content.split(' ')
			doc_freq    = dict( Counter(content) )

			for word in doc_freq:
				word_id    = dictionary.get(word, -1)
				if word_id != -1:
					word_count = doc_freq[word]
					frequencies.append( (word_id, document.document_id, word_count) )

			print "Loading doc " + str(document.document_id)
			query = 'INSERT INTO doc_term_freq VALUES (?,?,?)'
			cursor2.executemany(query, frequencies)

		self.connection.commit()
		logging.info("\nDone")


	def _compute_term_frequencies(self, document):
		""" Compute the frequency of each word in the given document.
		- The document is a list of tokens """

		return Counter(document).most_common()


	def save_wiki_vectors(self, vectors):
		""" Saves the wiki dictionary and vectors into the database """

		logging.info("\nSaving inverted_index into ESA database ...")
		self._save_inverted_index(vectors)


	def save_inverted_index(self, inverted_index):
		""" Saves the inverted index into the database """

		logging.info("Pruning index ...")
		values = []
		for term in inverted_index:

			if term == 33796:
				print "Number of documents related with beer: " + str(len(inverted_index[term]))

			# Discard insignificant associations
			documents = self.sliding_window_prunning(inverted_index[term])
			compacted = self.pack_vectors(documents)
			values.append( (term, compacted) )

		logging.info("Saving inverted index ...")
		# Clean table
		cursor = self.connection.cursor()
		query  = "DROP TABLE IF EXISTS term"
		cursor.execute(query)

		query  = "CREATE TABLE term (term_id INTEGER PRIMARY KEY, term_vector text)"
		cursor.execute(query)

		# Save inverted index
		self.connection.text_factory = str
		cursor = self.connection.cursor()
		query  = 'INSERT INTO term VALUES (?, ?)'
		cursor.executemany(query, values)
		self.connection.commit()


	def save_pruned_index(self, inverted_index):
		""" Self inverted_index once pruned with window_size=50 """

		logging.info("Saving inverted index ...")
		# Clean table
		cursor = self.connection.cursor()
		query  = "DROP TABLE IF EXISTS inverted_index"
		cursor.execute(query)

		query  = "CREATE TABLE inverted_index (term text, term_vector text)"
		cursor.execute(query)

		# Save inverted index
		self.connection.text_factory = str
		cursor = self.connection.cursor()
		query  = 'INSERT INTO inverted_index VALUES (?, ?)'
		cursor.executemany(query, inverted_index)
		self.connection.commit()


	def _save_inverted_index(self, vectors):
		""" Creates the inverted index based on the tf-idf vectors """

		inverted_index = defaultdict(list)

		# Clean table
		cursor = self.connection.cursor()
		query  = "DROP TABLE IF EXISTS term"
		cursor.execute(query)

		query  = "CREATE TABLE term (term_id INTEGER PRIMARY KEY, term_vector text)"
		cursor.execute(query)

		logging.info("Saving inverted index ...")

		for document_id, document_vector in vectors:

			for term_id, value in enumerate(document_vector):
				inverted_index[term_id].append( (document_id, value) )

		values = []
		for term in inverted_index:

			# Discard insignificant associations
			documents = self.sliding_window_prunning(inverted_index[term])
			compacted = self.pack_vectors(documents)
			values.append( (term, compacted) )

		# Save inverted index
		cursor = self.connection.cursor()
		query  = 'INSERT INTO term VALUES (?, ?)'
		cursor.executemany(query, values)
		self.connection.commit()


	def get_inverted_index(self, inverted_index):
		""" Returns the inverted index from the ESA database """

		logging.info("Retrieving the inverted index")

		cursor = self.connection.cursor()
		query  = 'SELECT term, term_vector FROM term tv LEFT JOIN term_map tm ON tv.term_id = tm.term_id'
		self.connection.text_factory = str
		cursor.execute(query)

		for term, term_vector in cursor:
			# Unpack binary string to vector form (8 comes from 'if' format)
			for i in xrange(0, len(term_vector), 8):
				document_id, value   = struct.unpack('if', term_vector[i:i+8])
				inverted_index[term].append( (document_id, value) )



	def get_pruned_inverted_index(self, inverted_index):

		logging.info("Retrieving the inverted index")

		cursor = self.connection.cursor()
		query  = 'SELECT term, term_vector FROM inverted_index'
		self.connection.text_factory = str
		cursor.execute(query)

		for term, term_vector in cursor:
			# Unpack binary string to vector form (8 comes from 'if' format)
			for i in xrange(0, len(term_vector), 8):
				document_id, value   = struct.unpack('if', term_vector[i:i+8])
				inverted_index[unicode(term)].append( (document_id, value) )



	def _save_term(self, term_id, documents):
		""" Saves the term with the list of documents assosiated with it """

		# Discard insignificant associations
		documents = self.sliding_window_prunning(documents)

		# Save the inverted index in the database
		cursor = self.connection.cursor()
		query  = 'INSERT INTO term VALUES (?, ?)'
		values = (term_id, self.pack_vectors(documents) )

		cursor.execute(query, values)
		self.connection.commit()


	def get_wiki_terms_frequencies(self):
		""" Returns the frequency values for each term in the Wikipedia corpus """

		result = {}
		cursor = self.connection.cursor()
		query  = 'SELECT term_id, doc_id, freq FROM doc_term_freq ORDER BY term_id, freq'

		for term_id, doc_id, freq in cursor.execute(query):
			result[ (term_id, doc_id) ] = freq

		return result


	def save_wiki_inverse_document_frequencies(self, number_of_documents):
		""" Saves the inverse document frequency values for the Wikipedia corpus """

		values     = []
		append     = values.append
		term_count = {}

		# Get corpus from database
		corpus = WikiCorpus(self.connection, self.testing)

		# Make it faster using the dictionary.dfs, which returns a dictionary with
		# the document frequency of each term.
		for document in corpus:
			content = set(document.content.split(' '))

			for word in content:
				term_count[word] = term_count.get(word, 0) + 1

			print "Doc " + str(document.document_id)

		for word in term_count:
			append( (word, math.log( float(number_of_documents) / term_count[word] )) )


		# Clean table 
		logging.info('Cleaning table ...')
		cursor = self.connection.cursor()
		query  = "DROP TABLE IF EXISTS inv_doc_freq"
		cursor.execute(query)

		query  = "CREATE TABLE inv_doc_freq (term text PRIMARY KEY, value real)"
		cursor.execute(query)

		logging.info('Inserting values ...')

		query  = "INSERT INTO inv_doc_freq  VALUES (?, ?)"
		cursor.executemany(query,values)
		self.connection.commit()



	def get_wiki_inverse_document_frequencies(self, inv_doc_freq):
		""" Returns the inverse document frequency values for the Wikipedia corpus """

		cursor = self.connection.cursor()
		query  = 'SELECT term, value FROM inv_doc_freq'
		cursor.execute(query)

		for term, value in cursor:
			inv_doc_freq[term] = value


	###############################################################################
	# Additional methods applied to TF-IDF Vectors
	###############################################################################

	@staticmethod
	def pack_vectors(vectors):
		""" Converts a list of (int, float) tuples into a binary string """

		result = []
		append = result.append

		for vector in vectors:
			#Pack: i for int (size 4), f for float (size 4) --> total size = 8
			append( struct.pack('if', *vector) )

		return ''.join(result)


	def sliding_window_prunning(self, vectors, window_size=100, threshold=0.05):
		""" Applies the algorithm that eliminate spurious associations
		between articles (concepts) and words. See paper for reference. """

		# Sort all concepts by the TF-IDF value in decreasing order
		# @vectors is a list of tuples, the second element is the value
		sorted_list = sorted(vectors, key=itemgetter(1))
		sorted_list = sorted_list[::-1] # Decreasing order

		# Scan the sequence with a @window_size of 100, and truncate the
		# sequence when the difference in scores between the first and last 
		# concepts in the window drops below 5% of the highest-score
		max_score = sorted_list[0][1]
		result    = []
		append    = result.append

		for i, (document_id, value) in enumerate(sorted_list):
			if len(result) >= window_size:
				# First and last elements of the window
				window_first = sorted_list[ max(0, i - window_size) ]
				window_last  = sorted_list[ max(0, i - 1) ]

				# Subtract elements' values
				scores_dif = window_first[1] - window_last[1]

				# Check prunning condition
				if scores_dif < max_score * threshold:
					break

			append( (document_id, value) )

		return result



	###############################################################################
	# Manage similarities results in database
	###############################################################################

	def create_clean_similarities_table(self, clean=True):
		""" Createsa new table of similarities """

		print self.setting['theme']

		cursor = self.connection.cursor()

		# Drop table if required
		if clean:
			query = 'DROP TABLE IF EXISTS esa_similarities_' + self.setting['theme']
			cursor.execute(query)

		query = 'CREATE TABLE IF NOT EXISTS esa_similarities_' + self.setting['theme'] + \
				' (question_id int, answer_id int, similarity real)'
		cursor.execute(query)
		self.connection.commit()


	def save_similarities(self, similarities):
		""" Saves the similarities into the ESA database """

		cursor = self.connection.cursor()
		query  = 'INSERT INTO esa_similarities_' + self.setting['theme'] + ' VALUES (?, ?, ?)'
		values = []

		for question_id, answer_id, sim in similarities:
			values.append( (question_id, answer_id, sim) )

		cursor.executemany(query, values)
		self.connection.commit()


	def create_clean_concept_doc_relation(self, clean=True):
		""" Creates a clean table if required """

		cursor = self.connection.cursor()

		# Drop table if required
		if clean:
			query = 'DROP TABLE IF EXISTS concept_doc_relatedness_' + self.setting['theme']
			cursor.execute(query)

		query = 'CREATE TABLE concept_doc_relatedness_' + self.setting['theme'] + \
				' (concept_id int, doc_id int, value real, type int)'
		cursor.execute(query)
		self.connection.commit()
		"""
		query = 'CREATE INDEX c_d_rel ON concept_doc_relatedness_' + self.setting['theme'] + \
			' (concept_id, doc_id)'
		cursor.execute(query)
		"""

	def save_concept_doc_relation(self, document_id, interpretation, concepts, type):
		""" For a given document, saves the related Wikipedia concepts and the values.
		Type indicates the type of document: question (1) or answer (2)  """

		cursor = self.connection.cursor()
		query  = 'INSERT INTO concept_doc_relatedness_' + self.setting['theme'] + ' VALUES (?, ?, ?, ?)'
		values = []

		for concept_id in concepts:
			values.append( (concept_id, document_id, interpretation[concept_id], type) )

		cursor.executemany(query, values)
		self.connection.commit()


	def write_relatedness_to_file(self, file_name):
		""" Writes in a file the related concepts found for each stack question/answer """

		logging.info("Collecting questions-concepts similarities ...")
		content = []

		esa_cursor = self.connection.cursor()
		query  = 'SELECT concept_id, doc_id, value FROM concept_doc_relatedness_' + self.setting['theme'] + ' WHERE type = 1 ORDER BY doc_id, value DESC'
		esa_cursor.execute(query)


		stack_connection = sqlite3.connect(self.setting['stack_dbpath'])
		wiki_connection  = sqlite3.connect(self.setting['wiki_dbpath'])
		stack_cursor     = stack_connection.cursor()
		wiki_cursor      = wiki_connection.cursor()

		for concept_id, question_id, value in esa_cursor:

			wiki_query   = 'SELECT title FROM wiki_articles WHERE id = ?'
			wiki_cursor.execute(wiki_query, (concept_id,))
			concept_name = wiki_cursor.fetchone()[0]

			if concept_name is None:
				concept_name = "NOT FOUND"

			stack_query = 'SELECT title, body FROM question WHERE id = ?'
			stack_cursor.execute(stack_query, (question_id,))
			question_title, question_body = stack_cursor.fetchone()

			if question_title is None:
				question_title = "NOT FOUND"
			

			content.append( (question_id, question_title, question_body, concept_id, concept_name, value) )


		# Write to file
		logging.info("Saving them in a file ...")
		with open(file_name, 'a') as f:
			current_id = 0
			for (question_id, question_title, question_body, concept_id, concept_name, value) in content:
				if current_id != question_id:
					f.write("\nNEW QUESTION \n")
					f.write(str(question_id) + ": " + question_title + '\n')
					f.write(question_body)
				current_id = question_id

				if current_id == question_id:
					f.write(str(concept_id) + "   " + str(value) + ": " + concept_name)



	###############################################################################
	# Methods used to calculate statistics
	###############################################################################

	def load_similarities_for_question(self, question_id, limit, ordered_ascending=True, order_by='similarity'):
		""" Given a question, returns its similarity values for every answer """

		cursor = self.connection.cursor()
		table  = 'esa_similarities_' + self.setting['theme']
		query  = ""

		if ordered_ascending:
			query = 'SELECT answer_id, similarity FROM ' + table + \
					' WHERE question_id = ? ORDER BY ' + order_by + ' ASC'
		else:
			query = 'SELECT answer_id, similarity FROM ' + table + \
					' WHERE question_id = ? ORDER BY ' + order_by + ' DESC'

		if limit != -1:
			query += ' LIMIT ' + str(limit)


		cursor.execute(query, (question_id,))

		return cursor.fetchall()

	def load_similar_users_to_question(self, question_id, limit, ordered_ascending=True, order_by='similarity'):
		""" Given a question id, returns its similarity values for every active user """

		cursor = self.connection.cursor()
		table  = 'esa_similarities_' + self.setting['theme']
		query  = ""

		if ordered_ascending:
			query = 'SELECT answer_id, similarity FROM ' + table + \
					' WHERE question_id = ? ORDER BY ' + order_by + ' ASC'
		else:
			query = 'SELECT answer_id, similarity FROM ' + table + \
					' WHERE question_id = ? ORDER BY ' + order_by + ' DESC'

		if limit != -1:
			query += ' LIMIT ' + str(limit)


		cursor.execute(query, (question_id,))

		return cursor.fetchall()

