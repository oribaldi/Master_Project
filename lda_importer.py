# -*- coding: utf-8 -*-
###########################################################################
#
# TUM Informatics
# Master Thesis Project: Indexing Methods for Social Search
# 
# Author: Oriana Baldizan
# Date:   December 2015
#
# lda_importer.py: Defines the LDA importer
#
# - Handles the LDA database
#
###########################################################################

# Python Modules
import logging
import sqlite3
import os
import sys


class LDAImporter (object):
	""" Manager for data used by the LDA algorithm """

	def __init__(self, setting):

		self.setting      = setting
		self.connection   = None


	def open_lda_db(self):

		self.connection = sqlite3.connect(self.setting['lda_dbpath'])

	def close_lda_db(self):

		self.connection.close()


	def create_clean_similarities_table(self, clean=True):
		""" Createsa new table of similarities """

		cursor = self.connection.cursor()

		# Drop table if required
		if clean:
			query = 'DROP TABLE IF EXISTS lda_similarities_' + self.setting['theme']
			cursor.execute(query)

		query = 'CREATE TABLE IF NOT EXISTS lda_similarities_' + self.setting['theme'] + \
				' (question_id int, answer_id int, similarity real)'
		cursor.execute(query)
		self.connection.commit()


	def save_similarities(self, similarities):
		""" Save questions/answers similarities to the database """

		cursor = self.connection.cursor()
		query  = 'INSERT INTO lda_similarities_' + self.setting['theme'] + ' VALUES (?, ?, ?)'
		values = []

		for question_id, answer_id, sim in similarities:
			values.append( (question_id, answer_id, sim) )

		cursor.executemany(query, values)
		self.connection.commit()


	def load_similarities_for_question(self, question_id, limit, ordered_ascending=True, order_by='similarity'):
		""" Given a question, returns its similarity values for every answer """

		cursor = self.connection.cursor()
		table  = 'lda_similarities_' + self.setting['theme']
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

		