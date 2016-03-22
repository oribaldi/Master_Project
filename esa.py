# -*- coding: utf-8 -*-
#####################################################################
#
# TUM Informatics
# Master Thesis Project: Indexing Methods for Social Search
# 
# Author: Oriana Baldizan
# Date:   October 2015
#
# esa.py : defines the ESA Method and the Vector Model using TF-IDF
#
# ESA - Explicit Semantic Analysis
#
#####################################################################

# Python Modules
from collections import defaultdict
from multiprocessing import Process, Pool
from functools import partial
from numpy.linalg import norm
from numpy import zeros, empty, dot
from operator import itemgetter
import itertools
import time
import logging
import sys
import math
import sqlite3

# Project Modules
from stack_importer import StackImporter, StackCorpus, StackUser
from wiki_preprocessor import WikiPreprocessor
from wiki_importer import WikiImporter
from esa_importer import ESAImporter, EsaIndex
from experiments import Experiments


class TfidfModel(object):
	""" Objects of this class represent the tfidf vectors
	of the Wikipedia articles """

	def __init__(self, dictionary, inv_doc_freq,):

		self.dictionary      = dictionary
		self.dictionary_size = len(dictionary)
		self.inv_doc_freq    = inv_doc_freq
		self.eps             = math.exp(-3)


	def __getitem__(self, document):

		vector  = zeros(self.dictionary_size)
		content = document.content.split(' ')
		doc_tf  = defaultdict(float)
		size    = 0 # length of the document

		# Faster than Counter
		for word in content:
			doc_tf[word] += 1.0
			size         += 1.0

		# Create tfidf vector
		for word in doc_tf:
			word_id = self.dictionary[word]
			tf      = doc_tf[word] / size # Normalized term frequency

			vector[word_id] = float(tf * self.inv_doc_freq[word])

		# Cosine normalization: divide each entry by the euclidean norm
		vector /= norm(vector)

		# Return only the pairs (word_id, tfidf value) with value > eps
		# The missing ones will be considered 0
		return [ (word_id, value) for word_id, value in enumerate(vector) if value > self.eps]


class ESA (object):
	""" ESA - Explicit Semantic Analysis """

	def __init__(self, setting):

		self.setting          = setting

		self.idf_values       = None

		self.wiki_corpus      = None
		self.wiki_dictionary  = None
		self.wiki_vectors     = []
		self.wiki_processor   = WikiPreprocessor(setting)
		self.wiki_importer    = WikiImporter(setting, self.wiki_processor)

		self.stack_corpus        = None
		self.answer_vectors      = {}
		self.question_vectors    = {}
		self.user_vectors        = {}
		self.user_content        = {}
		self.stack_importer      = StackImporter(setting)

		self.esa_importer        = ESAImporter(setting)
		self.inverted_index      = defaultdict(list)
		self.number_of_concepts  = 0

		self.experiments         = Experiments(setting)


	###############################################################################
	# Clean and load data
	###############################################################################
	def clean_and_load_data(self):
		""" Cleans the data and saves it in a database """

		self.wiki_importer.import_wiki_data()


	###############################################################################
	# Create and manage data used by ESA algorithm
	###############################################################################

	def build_esa_db(self):
		""" Initializes the ESA database """

		logging.info("\nCreating ESA database ...")

		self.esa_importer.open_esa_db()
	
		# Initialize database
		self.esa_importer.create_esa_db()

		# Save the dictionary and corpus of the Wikipedia data
		self.wiki_dictionary = self.wiki_importer.build_wiki_kb()

		# Save the inverse document frequencies in the ESA database
		number_of_documents = self.wiki_dictionary.num_docs #self.wiki_importer.get_number_of_concepts()
		self.esa_importer.save_wiki_inverse_document_frequencies(number_of_documents)

		self.esa_importer.close_esa_db()


	def load_esa_index(self):
		""" Gets the inverted index from the database """

		self.esa_importer.open_esa_db()

		self.esa_importer.get_pruned_inverted_index(self.inverted_index)
		logging.info("\nDone")

		self.esa_importer.close_esa_db()


	###############################################################################
	# Build TF-IDF Vectors
	###############################################################################

	def create_tf_idf_vectors(self):
		""" Creates them if not already in database """

		self.esa_importer.open_esa_db()

		# Calculate tfidf vectors for the Wikipedia articles
		self.create_tf_idf_wiki_vectors()

		# Save terms and vectors to ESA db
		#self.esa_importer.save_inverted_index(self.wiki_vectors)

		logging.info("\nDone")

		self.esa_importer.close_esa_db()


	def create_tf_idf_wiki_vectors(self):
		""" Keeping only non-zero entries of the vectors """

		wiki_corpus, self.wiki_dictionary = self.esa_importer.get_wiki_corpus_dictionary()
		
		logging.info("Retrieving idf values ...")
		inv_doc_freq = {}
		self.esa_importer.get_wiki_inverse_document_frequencies(inv_doc_freq)

		logging.info("Building the tfidf vectors and the inverse index ...")
		tfidf_model    = TfidfModel(self.wiki_dictionary, inv_doc_freq)
		inverted_index = defaultdict(list)

		for document in wiki_corpus:
			vector = tfidf_model[document]
			
			for term_id, value in vector:
				inverted_index[term_id].append( (document.document_id, value) )

			#print "Added " + str(document.document_id)
		
		logging.info("\n\tDone.")
		self.esa_importer.save_inverted_index(inverted_index)

		self.save_index_to_file(inverted_index)


	def _create_tf_idf_stack_vectors(self, only_questions=False):
		""" Create the tfidf vectors for the Stackexchange data. """

		# Load question and answer corpus
		logging.info("Loading stack corpus and dictionary ...")
		question_corpus = self.stack_importer.get_question_corpus()
		answer_corpus   = self.stack_importer.get_answer_corpus()

		corpus     = question_corpus + answer_corpus
		dictionary = self.stack_importer.get_dictionary_from_corpora([question_corpus, answer_corpus])
		dict_size  = len(dictionary)

		# Save stack dictionary
		stack_dict = {}
		for word_id, word in enumerate(dictionary.token2id):
			stack_dict[unicode(word)] = word_id

		self.idf_values = zeros(dict_size)

		logging.info("Determining question vectors ...")
		questions = StackCorpus(self.stack_importer.connection, "question")
		for question in questions:
			question_vector = zeros(dict_size)

			for word in question.body:
				word_id = stack_dict.get(unicode(word), -1)

				if word_id != -1:
					question_vector[word_id] = self.tf_idf(word, word_id, question.body, corpus)

			self.question_vectors[question.id] = question_vector

		logging.info("\n\tDone.")

		if only_questions: # Skip the answers
			return stack_dict

		logging.info("Determining answer vectors ...")
		answers   = StackCorpus(self.stack_importer.connection, "answer")
		
		for answer in answers:
			answer_vector = zeros(dict_size)

			for word in answer.body:
				word_id = stack_dict.get(unicode(word), -1)

				if word_id != -1:
					tf_idf = self.tf_idf(word, word_id, answer.body, corpus)
					answer_vector[word_id] = tf_idf

			self.answer_vectors[answer.id] = answer_vector

		logging.info("\n\tDone.")

		return stack_dict


	def _create_local_tf_idf_stack_vectors(self, user_id):
		""" Create the tfidf vectors for the local Stackexchange data of the given user """

		# Load question and answer corpus
		#logging.info("Loading stack corpus and dictionary ...")
		question_corpus = self.stack_importer.get_user_question_corpus(user_id)
		answer_corpus   = self.stack_importer.get_user_answer_corpus(user_id)

		corpus     = question_corpus + answer_corpus
		dictionary = self.stack_importer.get_dictionary_from_corpora([question_corpus, answer_corpus])
		dict_size  = len(dictionary)

		# Save stack dictionary
		stack_dict = {}
		for word_id, word in enumerate(dictionary.token2id):
			stack_dict[unicode(word)] = word_id

		self.idf_values = zeros(dict_size)

		#logging.info("Determining question vectors ...")
		questions = self.stack_importer.get_user_local_questions(user_id)

		for question in questions:
			question_vector = zeros(dict_size)

			for word in question.body:
				word_id = stack_dict.get(unicode(word), -1)

				if word_id != -1:
					question_vector[word_id] = self.tf_idf(word, word_id, question.body, corpus)

			self.question_vectors[question.id] = question_vector

		#logging.info("\n\tDone.")


		#logging.info("Determining answer vectors ...")
		answers = self.stack_importer.get_user_local_answers(user_id)

		for answer in answers:
			answer_vector = zeros(dict_size)

			for word in answer.body:
				word_id = stack_dict.get(unicode(word), -1)

				if word_id != -1:
					tf_idf = self.tf_idf(word, word_id, answer.body, corpus)
					answer_vector[word_id] = tf_idf

			self.answer_vectors[answer.id] = answer_vector

		#logging.info("\n\tDone.")

		return stack_dict


	def _create_user_tf_idf_stack_vector(self, user_id, stack_dict):
		""" Create the tfidf vector representation of a user, based on her answers"""
		
		aux = self.user_content.get(user_id, None)
		if aux is not None:
			return aux

		user_corpus = []
		user_words  = []
		answers = self.stack_importer.get_user_answers_to_questions(user_id)
		for answer in answers:
			user_corpus.append(answer.body)
			for word in answer.body:
				user_words.append(word)

		self.user_content[user_id] = user_words
		
		dict_size   = len(stack_dict)
		user_vector = zeros(dict_size)

		for word in set(user_words):
			word_id = stack_dict.get(unicode(word), -1)

			if word_id != -1:
				tf_idf = self.tf_idf(word, word_id, user_words, user_corpus)
				user_vector[word_id] = tf_idf

		self.user_vectors[user_id] = user_vector

		return user_words



	@staticmethod
	def tf(word, document):
		""" Returns the normalized frequency of the word in the given document """

		word_count = document.count(unicode(word))
		return float(word_count) / len(document)


	@staticmethod
	def df(word, corpus):
		""" Returns the number of documents in the collection that contain the given word """

		return sum(1 for document in corpus if unicode(word) in document)


	#@staticmethod
	def idf(self, word, corpus):
		""" Returns the inverse document frequency of the word in the documents collection """

		return math.log(len(corpus)) / self.df(word, corpus)


	def tf_idf(self, word, word_index, document, corpus):
		""" Returns the TF-IDF value for the given 
		word in the document of the corpus """

		# Calculate the term frequency value (tf)
		tf = self.tf(word, document)
		if tf == 0.0:
			return 0.0

		# Calculate the inverse document frequency value (idf)
		if self.idf_values[word_index] == 0.0:
			self.idf_values[word_index] = self.idf(word, corpus)

		return float(tf * self.idf_values[word_index])



	###############################################################################
	# Associations and Similarities of Stackexchange questions/answers using
	# Wikipedia's articles as concepts.
	###############################################################################

	def calculate_similarities(self):
		""" Applies the ESA algorithm to the global stack data """

		# Open database connections
		self.stack_importer.open_stack_db()
		self.esa_importer.open_esa_db()

		# Clean tables
		logging.info("Cleaning similarity tables ...")
		self.esa_importer.create_clean_concept_doc_relation()
		self.esa_importer.create_clean_similarities_table()

		logging.info("Loading the inverted index ...")
		self.esa_importer.get_pruned_inverted_index(self.inverted_index)

		#print "Has beer " + str(self.inverted_index.get(unicode("beer"), None))

		logging.info("Calculating stack tfidf vectors ...")
		stack_dictionary = self._create_tf_idf_stack_vectors()

		# For each question calculate similarity with each answer
		logging.info("\nCalculating questions-answers similarities ...")
		question_corpus = StackCorpus(self.stack_importer.connection, "question")
		
		for question in question_corpus:
			q_vector      = self.get_esa_vector(question.id, question.body, self.question_vectors[question.id], stack_dictionary, 1)
			q_vector_norm = norm(q_vector)
			similarities  = []

			answer_corpus = StackCorpus(self.stack_importer.connection, "answer")

			for answer in answer_corpus:
				a_vector  = self.get_esa_vector(answer.id, answer.body, self.answer_vectors[answer.id], stack_dictionary, 2)
				sim       = self.similarity(q_vector, q_vector_norm, a_vector)
				similarities.append( (question.id, answer.id, sim) )
			
			# Save similarities to databse
			logging.info("\nSaving similarities to database ...")
			self.esa_importer.save_similarities(similarities)

		self.esa_importer.close_esa_db()
		self.stack_importer.close_stack_db()

		logging.info("\nDone")


	def calculate_tf_idf_similarities(self):
		"""Applies the TF-IDF algorithm to the global stack data"""

		# Open database connections
		self.stack_importer.open_stack_db()
		self.esa_importer.open_esa_db()

		# Clean tables
		logging.info("Cleaning similarity tables ...")
		self.esa_importer.create_clean_similarities_table()

		logging.info("Calculating stack tfidf vectors ...")
		stack_dictionary = self._create_tf_idf_stack_vectors()

		# For each question calculate similarity with each answer
		question_corpus = StackCorpus(self.stack_importer.connection, "question")

		logging.info("\nCalculating questions-answers similarities ...")
		for question in question_corpus:
			q_vector      = self.question_vectors[question.id]
			q_vector_norm = norm(q_vector)
			similarities  = []

			answer_corpus = StackCorpus(self.stack_importer.connection, "answer")
			for answer in answer_corpus:
				a_vector  = self.answer_vectors[answer.id]
				sim       = self.similarity(q_vector, q_vector_norm, a_vector)
				similarities.append( (question.id, answer.id, sim) )
			
			# Save similarities to databse
			logging.info("\nSaving similarities to database ...")
			self.esa_importer.save_similarities(similarities)

		self.esa_importer.close_esa_db()
		self.stack_importer.close_stack_db()

		logging.info("\nDone")


	def calculate_local_tfidf_similarities(self):
		""" Applies TF-IDF to the local stack data, in order
		to calculate questions/answers similarities. The local
		data is measured per user.
		Returns the list of users that were filtered. """

		# Keep filtered users
		filtered_users = []

		# Open database connections
		self.esa_importer.open_esa_db()
		self.stack_importer.open_stack_db()

		# Clean similarity table
		self.esa_importer.create_clean_similarities_table()

		# For each question calculate its similarity with the all the answers given
		# by the users who answered the given question
		logging.info("Calculating questions/answers similarities ...")
		question_corpus = StackCorpus(self.stack_importer.connection, "question")

		for question in question_corpus:

			print "Question " + str(question.id)
			similarities  = []

			# Get the users that gave an answer to the question
			users = self.stack_importer.get_users_from_question(question.id)
			print "Users that replied: " + str(len(users))

			# Calculate the similarities of question with all
			# answers from the given users (related or not to question)
			for user_id in users:
				user_answers = self.stack_importer.get_user_answers_to_questions(user_id)

				# Only consider users with more than 1 answer
				if len(user_answers) > 5:

					print "User " + str(user_id)
					a = []
					for answer in user_answers:
						a.append(answer.id)
					print a

					# Calculate tf_idf vectors for the given user
					self.question_vectors.clear()
					self.answer_vectors.clear()
					stack_dictionary = self._create_local_tf_idf_stack_vectors(user_id)

					q_vector      = self.question_vectors[question.id]
					q_vector_norm = norm(q_vector)

					for answer in user_answers:
						a_vector = self.answer_vectors[answer.id]
						sim      = self.similarity(q_vector, q_vector_norm, a_vector)
						similarities.append( (question.id, answer.id, sim) )

				else:
					filtered_users.append(user_id)


			# Save similarities to databse
			logging.info("\nSaving similarities to database ...")
			self.esa_importer.save_similarities(similarities)

		# Close database connections
		self.esa_importer.close_esa_db()
		self.stack_importer.close_stack_db()
		logging.info("\nDone")

		return filtered_users


	def calculate_local_esa_similarities(self):
		""" Applies the ESA algorithm to the local stack data.
		This local data is measured per user. Returns the list
		of filtered users """

		# Keep filtered users
		filtered_users = []

		# Open database connections
		self.stack_importer.open_stack_db()
		self.esa_importer.open_esa_db()

		# Clean tables
		logging.info("Cleaning similarity tables ...")
		#self.esa_importer.create_clean_concept_doc_relation()
		self.esa_importer.create_clean_similarities_table()

		logging.info("Loading the inverted index ...")
		self.esa_importer.get_pruned_inverted_index(self.inverted_index)

		# For each question calculate its similarity with all the answers given
		# by the users who answered the given question
		logging.info("Calculating questions/answers similarities ...")
		question_corpus = StackCorpus(self.stack_importer.connection, "question")

		for question in question_corpus:

			print "Question " + str(question.id)
			similarities  = []

			# Get the users that gave an answer to the question
			users = self.stack_importer.get_users_from_question(question.id)
			print "Users that replied: " + str(len(users))

			# Calculate the similarities of question with all
			# answers from the given users (related or not to question)
			for user_id in users:
				user_answers = self.stack_importer.get_user_answers_to_questions(user_id)

				# Only consider users with more than 5 answers
				if len(user_answers) > 5:
					print "User " + str(user_id)

					# Calculate tf_idf vectors for the given user
					self.question_vectors.clear()
					self.answer_vectors.clear()
					stack_dictionary = self._create_local_tf_idf_stack_vectors(user_id)

					q_vector      = self.get_esa_vector(question.id, question.body, self.question_vectors[question.id], stack_dictionary, 1)
					q_vector_norm = norm(q_vector)

					for answer in user_answers:
						a_vector  = self.get_esa_vector(answer.id, answer.body, self.answer_vectors[answer.id], stack_dictionary, 2)
						sim       = self.similarity(q_vector, q_vector_norm, a_vector)
						similarities.append( (question.id, answer.id, sim) )

				else:
					filtered_users.append(user_id)


			# Save similarities to databse
			logging.info("\nSaving similarities to database ...")
			self.esa_importer.save_similarities(similarities)

		self.esa_importer.close_esa_db()
		self.stack_importer.close_stack_db()

		logging.info("\nDone")

		return filtered_users


	def get_esa_vector(self, id, document, tfidf_vector, dictionary, type):
		""" Creates the interpretation vector of the given document.
		- The document should be a set of tokens, already preprocessed
		- The vector represents the relatedness of the document
		with all the Wikipedia articles
		- Type indicates the type of document: question (1) or answer (2) """

		# Interpretation vector with dimensions = Wikipedia articles
		interpretation = zeros(2080905)

		for token in set(document):
			documents = self.inverted_index.get(unicode(token), None)
			word_id   = dictionary.get(unicode(token), -1)

			if documents is not None and word_id != -1:
				#print str(len(documents))
				for document_id, value in documents:
					interpretation[document_id] += (value * tfidf_vector[word_id])

		return interpretation


	def similarity(self, vector1, norm_vector1, vector2):
		""" Calculates the cosine similarity between the given vectors """

		# Cosine similartity
		sim = float(dot(vector1, vector2) / (norm_vector1 * norm(vector2)))
		return sim


	def save_relatedness_to_file(self, file_name):

		self.esa_importer.open_esa_db()
		self.esa_importer.write_relatedness_to_file(file_name)
		self.esa_importer.close_esa_db()



	### EXTRA ###
	def save_index_to_file(self, index=None, file_name='../data/ESA/index.txt'):

		index = defaultdict(list)

		# Extract it from DB
		self.esa_importer.open_esa_db()
		self.esa_importer.get_pruned_inverted_index(index)
		self.esa_importer.close_esa_db()

		# Copy to file
		logging.info("Saving them in a file ...")
		with open(file_name, 'a') as f:
			for word, doc_list in index.iteritems():
				#print word
				f.write(word + '\n')
				f.write(' '.join([str(x) for x in doc_list]))
				f.write('\n')


	def testing_beer_concept(self):

		tfidf_norm_values  = []
		tfidf_values       = []
		append_values      = tfidf_values.append
		append_norm_values = tfidf_norm_values.append

		self.esa_importer.open_esa_db()
		wiki_corpus, self.wiki_dictionary = self.esa_importer.get_wiki_corpus_dictionary()
		
		# IDF is fixed
		idf = 4.8225774331876625
		df  = 0

		for document in wiki_corpus:

			content = document.content.split(' ')

			if unicode("beer") in content:

				doc_tf  = defaultdict(float)
				size    = 0 # length of the document
				df     += 1

				# Faster than Counter
				for word in content:
					doc_tf[word] += 1.0
					size         += 1

				# Calculate tfidf value for word "beer" in Wiki data
				norm_value = (doc_tf[unicode("beer")] / size) * idf
				value      = doc_tf[unicode("beer")] * idf

				append_values( (document.document_id, value) )
				append_norm_values( (document.document_id, norm_value) )

		print "DF : " + str(df)

		# Sort each list
		sorted_norm_values = sorted(tfidf_norm_values, key=itemgetter(1))
		sorted_norm_values = sorted_norm_values[::-1]
		sorted_values      = sorted(tfidf_values, key=itemgetter(1))
		sorted_values      = sorted_values[::-1]

		# Print top 10 in each list
		print "Normalized : "
		print ' , '.join([str(id) + " " + str(value) for id,value in sorted_norm_values])

		print "\nNot normalized"
		print ' , '.join([str(id) + " " + str(value) for id,value in sorted_values])

		self.esa_importer.close_esa_db()


	def prun_inverted_index(self):
		""" Prun the inverted index """

		self.esa_importer.open_esa_db()

		index  = EsaIndex(self.esa_importer.connection)
		result = []
		append = result.append

		for term, vector in index:
			append( (term, vector) )

		self.esa_importer.save_pruned_index(result)

		self.esa_importer.close_esa_db()


	###############################################################################
	# Find the right person
	# Then, following a naive strong tie strategy, we could check for each question
	# which other users would have been asked following two strategies: (a) based 
	# on the social network ties (the ones with strongest ties) and (b) based on 
	# the content similarity (which answer is most similar to the question using 
	# TF-IDF or ESA, whatever you like best). Finally, we can compare both results 
	# with the ground truth (which users got actually asked in the dataset).
	###############################################################################

	def calculate_esa_similarities_to_users(self):

		# Open database connections
		self.stack_importer.open_stack_db()
		self.esa_importer.open_esa_db()

		# Clean tables
		logging.info("Cleaning similarity tables ...")
		self.esa_importer.create_clean_similarities_table()

		logging.info("Loading the inverted index ...")
		self.esa_importer.get_pruned_inverted_index(self.inverted_index)

		logging.info("Calculating questions tfidf vectors ...")
		stack_dictionary = self._create_tf_idf_stack_vectors(only_questions=True)

		# For each question determine which other users would have been asked
		logging.info("Calculating questions/users similarities ...")
		question_corpus = StackCorpus(self.stack_importer.connection, "question")

		users = self.stack_importer.get_active_users()

		for question in question_corpus:
			print "Question " + str(question.id)
			q_vector      = self.get_esa_vector(question.id, question.body, self.question_vectors[question.id], stack_dictionary, 1)
			q_vector_norm = norm(q_vector)
			similarities  = []

			for user_id in users:
				user_body = self._create_user_tf_idf_stack_vector(user_id, stack_dictionary)
				u_vector  = self.get_esa_vector(user_id, user_body, self.user_vectors[user_id], stack_dictionary, 2)
				sim       = self.similarity(q_vector, q_vector_norm, u_vector)
				similarities.append( (question.id, user_id, sim) )

			# Save similarities to databse
			logging.info("\nSaving similarities to database ...")
			self.esa_importer.save_similarities(similarities)

		self.esa_importer.close_esa_db()
		self.stack_importer.close_stack_db()

		logging.info("\nDone")


	def calculate_tfidf_similarities_to_users(self):

		# Open database connections
		self.stack_importer.open_stack_db()
		self.esa_importer.open_esa_db()

		# Clean tables
		logging.info("Cleaning similarity tables ...")
		#self.esa_importer.create_clean_concept_doc_relation()
		self.esa_importer.create_clean_similarities_table()

		logging.info("Calculating questions tfidf vectors ...")
		stack_dictionary = self._create_tf_idf_stack_vectors(only_questions=True)

		# For each question determine which other users would have been asked
		logging.info("Calculating questions/users similarities ...")
		question_corpus = StackCorpus(self.stack_importer.connection, "question")

		users = self.stack_importer.get_active_users()
	
		for question in question_corpus:
			print "Question " + str(question.id)
			q_vector      = self.question_vectors[question.id]
			q_vector_norm = norm(q_vector)
			similarities  = []

			for user_id in users:
				user_body = self._create_user_tf_idf_stack_vector(user_id, stack_dictionary)
				u_vector  = self.user_vectors[user_id]
				sim       = self.similarity(q_vector, q_vector_norm, u_vector)
				similarities.append( (question.id, user_id, sim) )

			# Save similarities to databse
			logging.info("\nSaving similarities to database ...")
			self.esa_importer.save_similarities(similarities)

		self.esa_importer.close_esa_db()
		self.stack_importer.close_stack_db()

		logging.info("\nDone")




	###############################################################################
	# Experiments - Calculate statistics on the data
	###############################################################################
	def initialize_experiments(self):

		self.experiments.open_experiment_db()
		self.experiments.create_experiments_db()
		self.experiments.close_experiment_db()

	def run_experiment_1(self):

		self.experiments.open_experiment_db()
		self.experiments.run_experiment_1(True)
		self.experiments.close_experiment_db()


	def run_experiment_1_avg(self, experiment_type='1_avg', algorithm='esa'):

		self.experiments.open_experiment_db()
		self.esa_importer.open_esa_db()
		self.stack_importer.open_stack_db()

		total_answers = self.stack_importer.get_number_of_answers()

		# Get number of answers for each question
		number_of_answers = self.stack_importer.get_number_of_original_answers()

		# Load similarities for each question
		logging.info("Loading similarities ...")
		question_corpus  = StackCorpus(self.stack_importer.connection, "question")
		similar_answers  = {}
		original_answers = {}
		
		for question in question_corpus:
			original_answers[question.id] = self.stack_importer.get_question_original_answers(question.id)
			similar_answers[question.id]  = self.esa_importer.load_similarities_for_question(question.id, -1, False)

		self.stack_importer.close_stack_db()
		self.esa_importer.close_esa_db()


		# Calculate avg precision and recall for each case
		precision = {}
		recall    = {}
		for limit in xrange(1,total_answers+1):
			logging.info("Calculating with limit %s", str(limit))

			avg_precision, avg_recall = self.experiments.run_experiment_1_avg(number_of_answers,
				original_answers, similar_answers, experiment_type, limit)
			precision[limit] = avg_precision
			recall[limit]    = avg_recall

		# Save into the database
		self.experiments.save_experiment_results(experiment_type, precision, recall)

		# Write them in a file
		folder = self.setting["experiments_folder"] + experiment_type + '_' + algorithm + '.dat'
		self.experiments.write_pr_curve(experiment_type, folder)

		self.experiments.close_experiment_db()

		logging.info("\nDone")


	def run_experiment_2_avg(self, algorithm='esa'):
		""" Same as run_experiment_1_avg but similarities were 
		calculated with local data per user """

		self.run_experiment_1_avg('2_avg', algorithm)


	def run_experiment_3_avg(self, algorithm='esa', experiment_type='3_avg'):
		""" Similar to experiment_1, but checking users instead of answers """

		self.experiments.open_experiment_db()
		self.esa_importer.open_esa_db()
		self.stack_importer.open_stack_db()

		# Get the number of active users
		active_users = len(self.stack_importer.get_active_users())

		# Get the users that gave an answer to each question
		asked_users = self.stack_importer.get_original_users()

		# Load similarities for each question
		logging.info("Loading similarities ...")
		question_corpus  = StackCorpus(self.stack_importer.connection, "question")
		similar_users  = {}
		original_users = {}

		for question in question_corpus:

			aux = asked_users.get(question.id, None)
			if aux is not None:
				original_users[question.id] = aux
				similar_users[question.id]  = self.esa_importer.load_similarities_for_question(question.id, -1, False)

		self.stack_importer.close_stack_db()
		self.esa_importer.close_esa_db()


		# Calculate avg precision and recall for each case
		precision = {}
		recall    = {}
		for limit in xrange(1,active_users+1):
			#print "Calculating with limit " + str(limit)
			logging.info("Calculating with limit %s", str(limit))

			avg_precision, avg_recall = self.experiments.run_experiment_3_avg(asked_users,
				original_users, similar_users, experiment_type, limit)
			precision[limit] = avg_precision
			recall[limit]    = avg_recall

		# Save into the database
		self.experiments.save_experiment_results(experiment_type, precision, recall)

		# Write them in a file
		folder = self.setting["experiments_folder"] + experiment_type + '_' + algorithm + '.dat'
		self.experiments.write_pr_curve(experiment_type, folder)

		self.experiments.close_experiment_db()

		logging.info("\nDone")


		
	
