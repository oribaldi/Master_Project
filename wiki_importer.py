# -*- coding: utf-8 -*-
###########################################################################
#
# TUM Informatics
# Master Thesis Project: Indexing Methods for Social Search
# 
# Author: Oriana Baldizan
# Date:   October 2015
#
# wiki_importer.py: Defines the methods that manage the data with SQLite
#
# - Creates a database for the data
# - Manipulates and retrieves the data
#
###########################################################################

# Python Modules
from gensim import corpora
from collections import Counter
from multiprocessing import Process, Pool

import sqlite3
import os.path
import logging
import re

# Project Modules
from wiki_preprocessor import WikiPreprocessor, WikiDocument


# Function for multiprocessing
def create_clean_document(raw_document):
	""" Preprocess a raw Wikipedia article """

	preprocessor   = WikiPreprocessor("nothing")
	clean_document = preprocessor.create_clean_document(raw_document)
	return clean_document


class WikiCorpus(object):
	""" Loads a document of the corpus one at a time """

	def __init__(self, connection):

		self.connection = connection

	def __iter__(self):

		# Open database connection
		cursor = self.connection.cursor()

		# Retrieve articles content
		query = 'SELECT id, content FROM wiki_articles ORDER BY id'
		cursor.execute(query)

		for document_id, content in cursor:
			yield WikiDocument( (document_id, content), not_raw=True )


class WikiImporter (object):
	""" Manager for Wikipedia data """

	def __init__(self, setting, preprocessor):

		self.setting      = setting
		self.connection   = None
		self.preprocessor = preprocessor

	def import_wiki_data(self):

		#self._create_wiki_db()
		self._import_wiki_articles()


	def get_number_of_concepts(self):

		# Database connection - instance variables
		connection = sqlite3.connect(self.setting['wiki_dbpath'])
		cursor     = connection.cursor()

		query    = 'SELECT COUNT(id) FROM wiki_articles'
		cursor.execute(query)
		articles = cursor.fetchone()

		connection.close()

		return articles[0]


	def _create_wiki_db(self):
		""" Creates the database for the Wikipedia articles """

		# Open database connection
		connection = sqlite3.connect(self.setting['wiki_dbpath'])

		logging.info("Creating wiki database ...")

		# Get cursor
		cursor = connection.cursor()

		# Create wiki table
		query = "DROP TABLE IF EXISTS wiki_articles"
		cursor.execute(query)

		query = 'CREATE TABLE wiki_articles (id INTEGER PRIMARY KEY, '\
				'doc_id int, title text, url text, content text)'
		cursor.execute(query)

		connection.commit()

		# Close connection
		connection.close()

	def _import_wiki_articles(self):
		""" Loads the Wikipedia data into the database """

		logging.info("Loading Wiki articles to db ...")

		# Open database connection
		connection = sqlite3.connect(self.setting['wiki_dbpath'], timeout=40.0)

		clean_counter = 0 # Number of documents after cleaning the data
		final_counter = 0 # Number of documents after removing small ones

		# Read all the files in the 'wiki_folder'
		for i in range(0,16):
			file_name      = self.setting['wiki_folder'] + "wiki_" + str(i)
			file_documents = []

			new_document   = True
			document_data  = []

			with open(file_name, 'r') as f:
				for line in f:

					line = line.decode('utf-8')

					if new_document:
						document_data = []
						new_document  = False

					if line.startswith("</doc>"):
						document     = WikiDocument(document_data)
						new_document = True
						file_documents.append(document)

					else:
						document_data.append(line)

			print "Number of documents " + str(len(file_documents))
						
			# Clean the data using different processes
			logging.info("Distributing work ...")
			p_pool = Pool(processes=5)
			clean_documents = p_pool.imap(create_clean_document, file_documents)
			p_pool.close()
			p_pool.join()

			# Remove documents with less than 100 words
			final_documents = []
			for doc in clean_documents:
				if len(doc.content.split(' ')) >= 100:
					final_documents.append(doc)
				clean_counter += 1

			final_counter += len(final_documents)

			logging.info("Inserting clean data ...")
			self._insert_articles_to_db(final_documents, connection)

			logging.info("Finished file ...")

		# Close connection
		connection.close()

		print "Clean documents " + str(clean_counter)
		print "Final documents " + str(final_counter)
		logging.info("\n\tDone.")



	def _insert_articles_to_db(self, documents, connection):
		""" Insert the list of articles to the database """ 

		# Get cursor
		cursor = connection.cursor()

		values = []

		for doc in documents:
			values.append( (doc.document_id, doc.title, doc.url, doc.content) )
			
		query  = 'INSERT INTO wiki_articles VALUES (NULL, ?, ?, ?, ?)'
		cursor.executemany(query, values)
		connection.commit()


	def _remove_small_documents(self, documents):
		""" Remove documents with less than 100 words """

		result = []

		for doc in documents:
			if len(doc.content) >= 100:
				result.append(doc)

		return result


	def store_wiki_tf_idf(self, tf_idf_vectors):
		""" Stores the tf_idf_vector into the database """

		# Open database connection
		connection = sqlite3.connect(self.setting['wiki_dbpath'])
		cursor     = connection.cursor()

		# Store vector as string
		#index = 
		#for vector in tf_idf_vectors:

			# Convert to string

			#query  = 'UPDATE wiki_articles SET tfidf=? WHERE id=?'
			#values = []
			#cursor.execute(query, values)
			#connection.commit()

		# Close database connection
		connection.close()

	
	def build_wiki_kb(self):
		""" Creates the Wikipedia's dictionary and corpus and save them in files """

		logging.info("Retrieving corpus ...")

		# Open database connection
		connection = sqlite3.connect(self.setting['wiki_dbpath'])
		corpus     = WikiCorpus(connection)

		logging.info("\nRetrieving dictionary ...")
		dictionary = corpora.Dictionary( document.content.split(' ') for document in corpus )

		logging.info("Removing very common and very uncommon words...")
		no_below = self.setting['filter_less_than_no_of_documents']
		no_above = self.setting['filter_more_than_fraction_of_documents']
		dictionary.filter_extremes(no_below=no_below, no_above=no_above)

		dictionary.compactify() # remove gaps in id sequence after words that were removed
		print dictionary
		self._save_dictionary(dictionary)

		logging.info("\nDone")

		connection.close()


	def _save_dictionary(self, dictionary):
		""" Saves the dictionary of terms """

		values = []

		for word_index, word in enumerate(dictionary.token2id):
			values.append( (word, word_index) )

		# Open database connection
		connection = sqlite3.connect(self.setting['esa_dbpath'])
		cursor     = connection.cursor()

		query      = 'INSERT INTO term_map VALUES (?, ?)'
		cursor.executemany(query, values)

		connection.commit()
		connection.close()



	### IGNORE ###

	def get_corpus(self, corpus):
		""" Returns the Wikipedia corpus """

		logging.info("Retrieving corpus ...")

		# Open database connection
		connection = sqlite3.connect(self.setting['wiki_dbpath'])
		cursor     = connection.cursor()

		# Retrieve articles content
		logging.info("\nRetrieving wikipedia corpus ...")
		query = 'SELECT id, content FROM wiki_articles ORDER BY id'
		cursor.execute(query)

		"""p_pool = Pool(processes=5)
		corpus = p_pool.imap(create_corpus, cursor.fetchall())
		p_pool.close()
		p_pool.join()
		#with open(self.setting['wiki_corpus'], 'a') as f:
		"""
		for document_id, content in cursor:
			corpus.append(content.split(' '))
			#f.write(str(document_id) + " " + content + '\n')

		connection.close()

	def get_wiki_tf_idf(self):
		""" Gets the Wikipedia tf-idf vectors from the database """

		corpus     = []
		vocabulary = None

		# Open database connection
		connection = sqlite3.connect(self.setting['wiki_dbpath'])
		cursor     = connection.cursor()

		# Retrieve articles content
		logging.info("\nRetrieving wikipedia corpus ...")
		query = 'SELECT id, content FROM wiki_articles'
		cursor.execute(query)

		for (id, document) in cursor:
			tokens = document.split(' ')
			corpus.append(tokens)

		# Close database connection
		connection.close()

		logging.info("\nRetrieving vocabulary ...")
		vocabulary = corpora.Dictionary(corpus)

		logging.info("Removing very common and very uncommon words...")
		no_below = self.setting['filter_less_than_no_of_documents']
		no_above = self.setting['filter_more_than_fraction_of_documents']
		self.vocabulary.filter_extremes(no_below=no_below, no_above=no_above)

		return vocabulary


	def get_stack_tf_idf(self):
		""" Gets the Stackoverflow tf-idf vectors from the database """

		# Open database connection
		connection = sqlite3.connect(self.setting['stack_dbpath'])
		cursor     = connection.cursor()

		# Retrieve vectors

		# Close database connection
		connection.close()