# -*- coding: utf-8 -*-
#############################################################
#
# TUM Informatics
# Master Thesis Project: Indexing Methods for Social Search
# 
# Author: Oriana Baldizan
# Date:   October 2015
#
# wiki_preprocessor.py: Defines the pre-processing methods
# of the wiki data
#
# - Removes stop words
# - Stemming
# - Removes very common and very uncommon words
#
#############################################################

# Python Modules
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora, utils

import logging
import unicodedata


# Global Variables
stop_words = stopwords.words("english")

wnl  = WordNetLemmatizer() # lematize(word)
stl  = LancasterStemmer()  # stem(word)
stp  = PorterStemmer()



class WikiDocument (object):
	""" Represents a Wikipedia article """

	def __init__(self, raw_data, not_raw=False):

		self.document_id = None
		self.content     = ""
		self.title       = ""
		self.url         = ""

		if not_raw:
			self._create_basic_document(raw_data[0], raw_data[1])
		else:
			self._create_document(raw_data)

	def _create_basic_document(self, id, content):

		self.document_id = id
		self.content     = content

	def _create_document(self, raw_data):

		for element in raw_data:

			if element.startswith("<doc"):
				# <doc id="46571441" url="https://en.wikipedia.org/wiki?curid=46571441" title="The bacterial, archaeal and plant plastid code">
				# Eliminar <doc, > y "
				element = element.replace("<doc ", "")
				element = element.replace("\"", "")
				element = element.replace(">", "")

				# Extract document id
				end              = element.find(" url=")
				self.document_id = element[3:end]

				# Extract document url
				start    = element.find("url=") + 4
				end      = element.find(" id=")
				self.url = element[start:end]

				# Extract document title
				start      = element.find("title=") + 6
				self.title = element[start:]

			elif element != "": # Ignore empty lines
				self.content += " " + element


class WikiPreprocessor (object):
	""" Preprocessor for Wikipedia data """

	def __init__(self, setting):

		self.setting    = setting
		self.corpus     = []
		self.vocabulary = corpora.Dictionary() #set


	def add_documents(self, documents):
		""" Add list of documents to vocabulary """

		self.vocabulary.add_documents(documents)


	def clean_documents(self, raw_documents):

		logging.info("Distributing work ...")
		p_pool = Pool()
		clean_documents = p_pool.imap(self._create_clean_document, raw_documents)
		p_pool.close()
		p_pool.join()

		return clean_documents

	def create_clean_document(self, document):

		#logging.info("\nCleaning document ...")
		clean_document = document
		clean_document.content = self.remove_special_characters(clean_document.content)

		# Convert document into tokens and clean it
		tokens = utils.simple_preprocess(clean_document.content)
		clean_content = self.remove_stop_words(tokens)
		clean_document.content = ' '.join(clean_content)

		return clean_document


	def remove_special_characters(self, text):
		""" Replaces the accents and special characters in text """

		nkfd_form = unicodedata.normalize('NFKD', unicode(text)).encode('ASCII', 'ignore')
		return nkfd_form


	def create_clean_corpus(self, raw_data):

		logging.info("\nCleaning data ...")
		for document in raw_data:
			# Convert document into tokens
			self.corpus.append( utils.simple_preprocess(document) )

		logging.info("\nRemoving short stop words ...")
		for (index, document) in enumerate(self.corpus):
			self.corpus[index] = self.remove_stop_words(document)

		logging.info("\nRetrieving vocabulary ...")
		self.vocabulary = corpora.Dictionary(self.corpus)


	@staticmethod
	def remove_stop_words(tokens):
		""" Removes stopwords and stems the rest of the words """

		result = []
		for token in tokens:
			if token not in stop_words:
				stem_token = stp.stem(token)
				if len(stem_token) > 2:
					result.append(stem_token)

		return result

	@staticmethod
	def get_dictionary_from_corpora(corpuses):

		combined_corpora = []
		for corpus in corpuses:
			combined_corpora += corpus

		return corpora.Dictionary(combined_corpora)