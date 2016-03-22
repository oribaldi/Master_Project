# -*- coding: utf-8 -*-
###########################################################################
#
# TUM Informatics
# Master Thesis Project: Indexing Methods for Social Search
# 
# Author: Oriana Baldizan
# Date:   October 2015
#
# stack_importer.py: Defines the methods that manage the 
#                    stack data with SQLite
#
# - Creates a database for the data
# - Manipulates and retrieves the data
#
###########################################################################

# Python Modules
from gensim import utils, corpora
from BeautifulSoup import *
from multiprocessing import Process, Pool
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from numpy import zeros, histogram, mean
from collections import defaultdict
import unicodedata

import sqlite3
import os.path
import logging
import re

# Project Modules
stop_words = stopwords.words("english")
stp        = PorterStemmer()


class StackPost (object):
	""" Represents a question/answer in the stack data """

	def __init__(self, id, body):

		self.body = body
		self.id   = id

		self._clean_body()


	def _clean_body(self):
		""" Preprocess the body of the post """

		tmp_text  = self.strip_code_blocks(self.body)
		tmp_text  = self.remove_special_characters(tmp_text)
		tokens    = utils.simple_preprocess(tmp_text)
		self.body = self.remove_short_words(tokens)


	def remove_special_characters(self, text):
		""" Replaces the accents and special characters in text """

		nkfd_form = unicodedata.normalize('NFKD', unicode(text)).encode('ASCII', 'ignore')
		#unic_form = u"".join([c for c in nkfd_form if not unicodedata.combining(c)])
		return nkfd_form


	@staticmethod
	def strip_code_blocks(content):

		result      = content
		soup        = BeautifulSoup(result)
		code_blocks = soup.findAll('code')

		for block in code_blocks:
			block.extract()

		return result


	@staticmethod
	def remove_short_words(tokens):

		result = []

		for token in tokens:
			if token not in stop_words and len(token) > 2:
				stem_token = stp.stem(token)
				result.append(stem_token)

		return result


class StackCorpus(object):
	""" Loads a document of the given corpus one at a time """

	def __init__(self, connection, table="question"):

		self.connection = connection
		self.table      = table

	def __iter__(self):

		# Open database connection
		cursor = self.connection.cursor()

		# Retrieve data
		if self.table == "question":
			query = 'SELECT id, title, body FROM question ORDER BY id'
			cursor.execute(query)

			for id, title, body in cursor:
				yield StackPost(id, ' '.join([title, body]))

		else:
			query = 'SELECT id, body FROM answer ORDER BY id'
			cursor.execute(query)

			for id, body in cursor:
				yield StackPost(id, body)


class StackUser(object):
	""" Loads a document of the given corpus one at a time """

	def __init__(self, connection):

		self.connection = connection

	def __iter__(self):

		# Open database connection
		cursor = self.connection.cursor()

		# Retrieve data
		query = 'SELECT id, elementId, displayName FROM user ORDER BY id'
		cursor.execute(query)

		for id, elementId, name in cursor:
			yield (elementId, name)



class StackImporter (object):
	""" Manager for Stackoverflow data """

	def __init__(self, setting):

		self.setting      = setting
		self.connection   = None
		self.preprocessor = None

		self.filtered_users = []

	
	def open_stack_db(self):

		self.connection = sqlite3.connect(self.setting['stack_dbpath'])

	def close_stack_db(self):

		# Close connection
		self.connection.close()


	###############################################################################
	# Data saving and retrieval
	###############################################################################

	def get_number_of_questions(self):

		# Database connection - instance variables
		self.connection = sqlite3.connect(self.setting['stack_dbpath'])
		self.cursor = self.connection.cursor()

		self.corpus = []

		sql = 'SELECT COUNT(id) FROM question'
		self.cursor.execute(sql)
		questions = self.cursor.fetchone()

		return questions[0]


	def get_number_of_answers(self):

		# Database connection - instance variables
		self.cursor = self.connection.cursor()

		self.corpus = []

		sql = 'SELECT COUNT(id) FROM answer'
		self.cursor.execute(sql)
		answers = self.cursor.fetchone()

		return answers[0]


	def get_question_corpus(self):

		corpus = []

		# Connect to database
		connection = sqlite3.connect(self.setting['stack_dbpath'])
		questions  = StackCorpus(connection)

		logging.info("Loading questions...")
		for question in questions:
			corpus.append(question.body)

		connection.close()

		return corpus


	def get_answer_corpus(self):

		corpus = []

		# Connect to database
		connection = sqlite3.connect(self.setting['stack_dbpath'])
		answers    = StackCorpus(connection, "answer")

		logging.info("Loading answers...")
		for answer in answers:
			corpus.append(answer.body)

		connection.close()

		return corpus

	@staticmethod
	def get_dictionary_from_corpora(corpuses):

		combined_corpora = []

		for corpus in corpuses:
			combined_corpora += corpus

		logging.info("Building the dictionary...")
		dictionary = corpora.Dictionary(combined_corpora)

		#logging.info("Removing very common and very uncommon words...")
		#no_below = self.setting['filter_less_than_no_of_documents']
		#no_above = self.setting['filter_more_than_fraction_of_documents']
		#dictionary.filter_extremes(no_below=no_below, no_above=no_above)

		return dictionary


	###############################################################################
	# Methods used to calculate statistics
	###############################################################################

	def get_question_original_answers(self, question_id, with_score=False):
		""" Returns the answers of the given question. 
		Returns answer raiting if with_score = True """

		answers = []

		cursor = self.connection.cursor()
		query  = 'SELECT id, score, ownerUserId FROM answer WHERE questionId = ? ORDER BY score DESC'
		cursor.execute(query, (question_id,))

		for id, score, user_id in cursor:

			if user_id is not None and user_id not in self.filtered_users:
				if with_score:
					answers.append( (id, score) )
				else:
					answers.append(id)

		return answers


	def get_number_of_original_answers(self):
		""" Returns the number of answers each question has in the original data.
		However, it ignores the answers where ownerUserId is None """

		cursor  = self.connection.cursor()
		answers = {}

		query   = 'SELECT a.questionId, COUNT(a.id) FROM answer a LEFT JOIN question q WHERE q.id = a.questionId GROUP BY a.questionId'
		cursor.execute(query)
		for question_id, number_answers in cursor:
			answers[question_id] = number_answers


		# Remove those answers where onwnerUserId is None
		query   = 'SELECT questionId, ownerUserId FROM answer'
		cursor.execute(query)
		for question_id, user_id in cursor:
			if user_id is None or user_id in self.filtered_users:
				answers[question_id] -= 1

		return answers


	def get_documents_from_ids(self, ids, type):
		""" Retrieves a list of Stackexchange documents that correspond to the given ids """

		cursor    = self.connection.cursor()
		query     = ''
		q_list    = ','.join(["?"]*len(ids))
		documents = []

		if type == 1:
			query = 'SELECT id, title, body FROM question WHERE id IN (' + q_list + ')'
			cursor.execute(query, ids)

			for id, title, body in cursor:
				documents.append( StackPost(id, ' '.join([title, body])) )

		else:
			query = 'SELECT id, body FROM answer WHERE id IN (' + q_list + ')'
			cursor.execute(query, ids)

			for id, body in cursor:
				documents.append( StackPost(id, body) )

		return documents


	def get_active_users(self):
		""" Returns the ids of the active users. A user is active if it has submitted at least one answer
		(only considering answers for now) """

		# Get connection
		cursor = self.connection.cursor()
		query  = 'SELECT DISTINCT ownerUserId FROM answer'
		cursor.execute(query)

		users = [id[0] for id in cursor if id[0] is not None]

		return users

	def get_all_users(self):
		""" Returns the ids of users with at least one question or answer """

		# Get connection
		cursor = self.connection.cursor()
		
		query  = 'SELECT DISTINCT ownerUserId FROM answer'
		cursor.execute(query)
		users  = [id[0] for id in cursor if id[0] is not None]

		query  = 'SELECT DISTINCT ownerUserId FROM question WHERE answerCount > 0'
		cursor.execute(query)
		users += [id[0] for id in cursor if id[0] is not None]

		return list(set(users))

	def get_original_users(self):
		""" Returns the number of users that answer a given question. """

		users  = {}
		cursor = self.connection.cursor()
		
		query  = 'SELECT DISTINCT questionId FROM answer'
		cursor.execute(query)
		for row in cursor:
			question_id = row[0]
			users[question_id] = self.get_users_from_question(question_id)

		return users


	###############################################################################
	# Methods used to recreate the local knowledge of each user in Stackexchange
	#
	# A user only has access to:
	# 1. The questions she received (i.e. the ones she answered)
	# 2. The questions she asked
	# 3. The answers she received on her questions
	# 4. The answers she gave
	#
	# A user can only use her individual corpus (from 1 to 4)
	###############################################################################

	#def question_avg_answers

	def user_avg_answers(self):
		""" Calculates the avg number of answers for the Stack users """

		# Get database connection
		cursor = self.connection.cursor()

		# Get number of users
		query  = 'SELECT id, elementId FROM user'
		cursor.execute(query)
		
		number_of_users = 0
		user_dict = {}
		for id, user_id in cursor:
			user_dict[user_id] = id
			number_of_users += 1

		print "Number of users " + str(number_of_users)

		users_answers     = defaultdict(int) #zeros(number_of_users+1)
		number_of_answers = 0

		# Get number of answers for each user
		query  = 'SELECT ownerUserId, id FROM answer'
		cursor.execute(query)

		for user_id, answer_id in cursor:
			if user_id is not None:
				id = user_dict[user_id]
				users_answers[id] += 1
				number_of_answers += 1


		user_histogram, bins = histogram(users_answers.values(), bins=10)
		print user_histogram
		print '\n'
		print bins
		print user_histogram.sum()

		users_no_answers = 0
		for e in users_answers:
			if e == 0:
				users_no_answers +=1

		print "Users without answers " + str(users_no_answers)
		print "Avg answers " + str(mean(users_answers.values()))


	def get_user_question_corpus(self, user_id):
		""" Returns the local question corpus of the given user. This includes
		the questions asked by the user and the questions answered by the user. """

		corpus = []
		append = corpus.append

		#logging.info("Loading local questions ...")
		user_questions   = self.get_user_asked_questions(user_id)
		user_a_questions = self.get_user_answered_questions(user_id)

		total_questions  = self._remove_duplicates(user_questions + user_a_questions)
		for question in total_questions:
			append(question.body)

		return corpus


	def get_user_answer_corpus(self, user_id):
		""" Returns the local answer corpus of the given user. This includes
		the answers the user has given and the answers to the questions she posted. """

		corpus = []
		append = corpus.append

		#logging.info("Loading local answers ...")
		user_answers   = self.get_user_answers_to_questions(user_id)
		user_questions = self.get_user_asked_questions(user_id)
		user_q_answers = self.get_answers_to_user_questions(user_id, user_questions)

		total_answers  = self._remove_duplicates(user_answers + user_q_answers)
		for answer in total_answers:
			append(answer.body)

		return corpus


	def _remove_duplicates(self, stack_posts):
		""" Remove duplicates from a list of StackPost objects """

		posts = {}
		for post in stack_posts:
			if posts.get(post.id, None) is None:
				posts[post.id] = post

		return posts.values()


	def get_user_local_questions(self, user_id):
		""" Returns the local questions of the given user. """

		user_questions   = self.get_user_asked_questions(user_id)
		user_a_questions = self.get_user_answered_questions(user_id)

		return self._remove_duplicates(user_questions + user_a_questions)


	def get_user_local_answers(self, user_id):
		""" Returns the local answers of the given user. """

		user_answers   = self.get_user_answers_to_questions(user_id)
		user_questions = self.get_user_asked_questions(user_id)
		user_q_answers = self.get_answers_to_user_questions(user_id, user_questions)

		return self._remove_duplicates(user_answers + user_q_answers)


	def get_user_answered_questions(self, user_id):
		""" Returns the list of questions the user has answered.
		- @answered_questions is a list StackPost objects """

		answered_questions = []
		append = answered_questions.append

		# Get the ids of the questions the user has answered
		cursor = self.connection.cursor()
		query  = 'SELECT questionId FROM answer WHERE ownerUserId = ?'
		cursor.execute(query, (user_id,))

		questions_ids = []
		for id in cursor:
			if id[0] not in questions_ids:
				questions_ids.append(id[0])

		ids_list      = ','.join(["?"]*len(questions_ids))

		# Get the content of the above questions
		query = 'SELECT id, title, body FROM question WHERE id IN (' + ids_list + ')'
		cursor.execute(query, questions_ids)

		for id, title, body in cursor:
			append( StackPost(id, ' '.join([title, body])) )

		return answered_questions


	def get_user_asked_questions(self, user_id):
		""" Returns the list of questions the user asked.
		- @questions is a list of StackPost objects """

		questions = []
		append    = questions.append

		cursor = self.connection.cursor()
		query  = 'SELECT id, title, body FROM question WHERE ownerUserId = ?'
		cursor.execute(query, (user_id,))

		for id, title, body in cursor:
			append( StackPost(id, ' '.join([title, body])) )

		return questions


	def get_user_answers_to_questions(self, user_id):
		""" Returns the list of user's answers to questions from other users.
		- @answers is a list of StackPost objects"""

		answers = []
		append = answers.append

		cursor = self.connection.cursor()
		query  = 'SELECT id, body FROM answer WHERE ownerUserId = ?'
		cursor.execute(query, (user_id,))

		for id, body in cursor:
			append( StackPost(id, body) )

		return answers


	def get_answers_to_user_questions(self, user_id, questions):
		""" Returns the list of answers given to the user's questions.
		- @questions is a list of StackPost objects """

		answers  = []
		append   = answers.append
		ids_list = ','.join(["?"]*len(questions))

		cursor   = self.connection.cursor()
		query    = 'SELECT id, body FROM answer WHERE questionId IN (' + ids_list + ')'
		values   = [question.id for question in questions]
		cursor.execute(query, values)

		for id, body in cursor:
			append( StackPost(id, body) )

		return answers


	def get_users_from_question(self, question_id):
		""" Returns the list of users that gave an answer to the given question """

		users  = []
		append = users.append

		cursor = self.connection.cursor()
		query  = 'SELECT ownerUserId FROM answer WHERE questionId = ?'
		cursor.execute(query, (question_id,))

		for row in cursor:
			user_id = row[0]
			if user_id not in users and user_id is not None:
				append(user_id)

		return users


	def get_user_friends(self, user_id):
		""" Returns the list of users the given @user_id is related to.
		The relation is based on the participation on the same posts """

		cursor = self.connection.cursor()
		questions = []
		friends   = []

		# Get the questions the user has asked
		query  = 'SELECT id FROM question WHERE ownerUserId = ?'
		cursor.execute(query, (user_id,))
		questions += [row[0] for row in cursor]

		# Get the users that answered the questions
		query  = 'SELECT DISTINCT ownerUserId FROM answer WHERE questionId = ?'
		for question_id in questions:
			cursor.execute(query, (question_id,))
			friends += [row[0] for row in cursor if row[0] is not None]

		# Get the questions the user has answered
		query  = 'SELECT questionId FROM answer WHERE ownerUserId = ?'
		cursor.execute(query, (user_id,))
		questions = []
		questions += [row[0] for row in cursor]

		# Get the users that posted those questions
		query  = 'SELECT DISTINCT ownerUserId FROM question WHERE id = ?'
		for question_id in questions:
			cursor.execute(query, (question_id,))
			friends += [row[0] for row in cursor if row[0] is not None]

		# Remove duplicates
		friends = list(set(friends))

		# Build the tuples for the edges
		edges = [(user_id, friend) for friend in friends]

		return edges

