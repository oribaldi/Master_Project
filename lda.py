# -*- coding: utf-8 -*-
###########################################################################
#
# TUM Informatics
# Master Thesis Project: Indexing Methods for Social Search
# 
# Author: Oriana Baldizan
# Date:   December 2015
#
# lda.py: Defines the topic model algorithm LDA
#
# - Only implementing LDA with local data
#
###########################################################################

# Python Modules
from gensim import models, utils
import logging
import sqlite3
import os
import sys
import tempfile

#tempfile.tempdir = 'C:\\Users\\Oriana\\Documents\\Univ\\TUM\\Thesis\\Experiments\\tmp\\'
tempfile.tempdir  = '../tmp/'

# Project Modules
from stack_importer import StackImporter, StackCorpus, StackUser
from experiments import Experiments
from lda_importer import LDAImporter

sys.path.append("../LDA-SO-master/")
import Metric


class LDA (object):
	""" LDA - Latent Dirichlet Porcesses """

	def __init__(self, setting):

		self.setting          = setting
		self.mallet_path      = setting['malletpath']
		self.number_of_topics = setting['nooftopics']
		self.number_of_iter   = setting['noofiterations']

		self.stack_importer   = StackImporter(setting)
		self.lda_importer     = LDAImporter(setting)
		self.experiments      = Experiments(setting)

		self.model            = None
		self.corpus           = None
		self.dictionary       = None
		self.answer_corpus    = None

		directory = self.setting['lda_folder']
		file_name = 'local_lda_model' + self.setting['theme'] + '.gs'
		self.path = ''.join([directory, file_name])


	def __iter__(self):

		for document in self.corpus:
			yield self.dictionary.doc2bow(document)



	def calculate_similarities(self):

		# Open database connections
		self.lda_importer.open_lda_db()
		self.stack_importer.open_stack_db()

		# Clean similarity table
		self.lda_importer.create_clean_similarities_table()

		self._learn_model()

		logging.info("Loading dictionary ...")
		self._load_dictionary()

		logging.info("Calculating questions/answers similarities ...")
		question_corpus = StackCorpus(self.stack_importer.connection, "question")

		for question in question_corpus:

			print "Question " + str(question.id)
			similarities  = []
			answer_corpus = StackCorpus(self.stack_importer.connection, "answer")

			# Get topics in the question
			bow = self.dictionary.doc2bow(question.body)
			question_topics = self.model[bow]

			for answer in answer_corpus:

				# Get topics in the answer
				bow = self.dictionary.doc2bow(answer.body)
				answer_topics = self.model[bow]

				# Similarities
				similarities.append((question.id, answer.id, self._compare_documents(question_topics, answer_topics)))

			# Save similarities to databse
			logging.info("\nSaving similarities to database ...")
			self.lda_importer.save_similarities(similarities)

		# Close database connections
		self.stack_importer.close_stack_db()
		self.lda_importer.close_lda_db()


	def _learn_model(self):
		self.model = models.wrappers.LdaMallet(self.mallet_path, corpus=self, num_topics=self.number_of_topics,
					id2word=self.dictionary, iterations=self.number_of_iter)


	def _load_dictionary(self):

		self.stack_importer.open_stack_db()

		# Load dictionary
		question_corpus = self.stack_importer.get_question_corpus()
		answer_corpus   = self.stack_importer.get_answer_corpus()
		corpus          = question_corpus + answer_corpus
		self.dictionary = self.stack_importer.get_dictionary_from_corpora([question_corpus, answer_corpus])

		self.stack_importer.close_stack_db()


	def run_experiment_1_avg(self, experiment_type='1_avg', algorithm='esa'):

		self.experiments.open_experiment_db()
		self.lda_importer.open_lda_db()
		self.stack_importer.open_stack_db()

		total_answers = self.stack_importer.get_number_of_answers()

		# Get number of answers for each question
		number_of_answers = self.stack_importer.get_number_of_original_answers()

		# Load similarities for each question
		logging.info("Loading similarities ...")
		question_corpus  = StackCorpus(self.stack_importer.connection, "question")
		similar_answers  = {}
		original_answers = {}
		#i = 0
		for question in question_corpus:

			#if i == 5:
			#	break
			#i += 1
			original_answers[question.id] = self.stack_importer.get_question_original_answers(question.id)
			similar_answers[question.id]  = self.esa_importer.load_similarities_for_question(question.id, -1, False)

		self.stack_importer.close_stack_db()
		self.lda_importer.close_lda_db()


		# Calculate avg precision and recall for each case
		precision = {}
		recall    = {}
		#total_answers = 20
		for limit in xrange(1,total_answers+1):
			#print "Calculating with limit " + str(limit)
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


	###############################################################################
	# Create the local model
	###############################################################################

	def calculate_local_similarities(self):
		""" Calculates similarities between local questions/answers.
			Returns the list of filtered users """

		# Keep filtered users
		filtered_users = []

		# Open database connections
		self.lda_importer.open_lda_db()
		self.stack_importer.open_stack_db()

		# Clean similarity table
		self.lda_importer.create_clean_similarities_table()

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

					self._learn_local_model(user_id)

					# Get topics in the question
					bow = self.dictionary.doc2bow(question.body)
					question_topics = self.model[bow]

					# Get topics in the answers and calculate similarities with current question
					for answer in user_answers:
						bow = self.dictionary.doc2bow(answer.body)
						answer_topics = self.model[bow]

						# Similarities
						similarities.append((question.id, answer.id, self._compare_documents(question_topics, answer_topics)))
				else:
					filtered_users.append(user_id)

			# Save similarities to databse
			logging.info("\nSaving similarities to database ...")
			self.lda_importer.save_similarities(similarities)
			#similarities.clear()


		# Close database connections
		self.stack_importer.close_stack_db()
		self.lda_importer.close_lda_db()

		return filtered_users


	def _learn_local_model(self, user_id):
		""" Learns the LDA model with local knowledge """

		# Load question and answer corpus
		question_corpus    = self.stack_importer.get_user_question_corpus(user_id)
		self.answer_corpus = self.stack_importer.get_user_answer_corpus(user_id)
		self.corpus        = question_corpus + self.answer_corpus
		self.dictionary    = self.stack_importer.get_dictionary_from_corpora([question_corpus, self.answer_corpus])

		# Create model
		self.model = models.wrappers.LdaMallet(self.mallet_path, corpus=self, num_topics=self.number_of_topics,
					id2word=self.dictionary, iterations=self.number_of_iter)


	@staticmethod
	def _compare_documents(document1, document2):
		""" Calculates the distance between the given documents """

		doc1_topic_description = []
		doc2_topic_description = []

		for (topic, weight) in document1:
			doc1_topic_description.append(weight)

		for (topic, weight) in document2:
			doc2_topic_description.append(weight)

		return Metric.js_distance(doc1_topic_description, doc2_topic_description)



	def run_experiment_2_avg(self, experiment_type='2_avg', algorithm='lda_local_2'):

		self.experiments.open_experiment_db()

		self.lda_importer.open_lda_db()
		self.stack_importer.open_stack_db()

		total_answers = self.stack_importer.get_number_of_answers()

		# Get number of answers for each question
		number_of_answers = self.stack_importer.get_number_of_original_answers()

		# Load similarities for each question
		logging.info("Loading similarities ...")
		question_corpus  = StackCorpus(self.stack_importer.connection, "question")
		similar_answers  = {}
		original_answers = {}
		#i = 0
		for question in question_corpus:

			#if i == 5:
			#	break
			#i += 1

			original_answers[question.id] = self.stack_importer.get_question_original_answers(question.id)
			similar_answers[question.id]  = self.lda_importer.load_similarities_for_question(question.id, -1, False)

		self.stack_importer.close_stack_db()
		self.lda_importer.close_lda_db()

		# Calculate avg precision and recall for each case
		precision = {}
		recall    = {}
		for limit in xrange(1,total_answers+1):
			print "Calculating with limit " + str(limit)

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