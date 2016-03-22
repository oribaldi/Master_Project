#############################################################
#
# TUM Informatics
# Master Thesis Project: Indexing Methods for Social Search
# 
# Author: Oriana Baldizan
# Date:   October 2015
# 4929936 wikipedia articles
# 4929500
# Final documents 4201775 (removed small articles)
# Now final: 2080904
# Number of words: 100000
#############################################################

# Python Modules
import logging
import os
import sys
import cProfile
import pstats

import sqlite3

# Proyect Modules
from esa import ESA
from lda import LDA


#############################################################


if len(sys.argv) > 1:
	theme = sys.argv[1]
"""
else:
	print('No theme provided')
	sys.exit(0)
"""

theme = "travel"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

setting = {

	'theme': theme,

	'wiki_folder': '../Datasets/Wikipedia/output/AA/',
	'stack_folder': '../Datasets/Stackoverflow/',
	
	'wiki_dbpath': '../data/Wiki/enwiki.db',
	'stack_dbpath': '../data/Stack/' + theme + '.db',
    'esa_dbpath': '../data/ESA/esa.db',
    'lda_dbpath': '../data/LDA/lda.db',
    'mini_esa_dbpath': '../data/ESA/mini_esa.db',
    'experiments_dbpath': '../data/Experiments/' + theme + '/experiments.db',

    'mini_test': False,

    'wiki_dict': '../data/Wiki/wiki.dict',
    'wiki_corpus': '../data/Wiki/wiki_corpus.txt',

    'wiki_loaded': False,
    'stack_loaded': False,

    # Normalize Wikipedia article vectors
    'cosine_norm': True,

    'folderprefix': '../data/',

    'lda_folder': '../data/LDA/',
    'experiments_folder': '../data/Experiments/' + theme + '/pr_curve_experiment_',

    'resultfolder': '../results/',
    'resultdbfolder': '../results/model/similarities.db',
    # 'resultfolder': '/srv/cordt-mt/results/',

    # Must escape the \ for making it work on Windows
    'malletpath': 'C:/Users/Oriana/Documents/Univ/TUM/Thesis/Experiments/mallet-2.0.7/bin/mallet',

    'nooftopics': 100,
    'noofwordsfortopic': 100,
    'noofiterations': 1000,
    'noofprocesses': 20,

    # Filter all words that appear in less documents
    'filter_less_than_no_of_documents': 5,

    # Filter all documents that appear in more than the given fraction of documents
    'filter_more_than_fraction_of_documents': 0.5,
}


if __name__ == '__main__':

    reload(sys)
    sys.setdefaultencoding("utf-8")

    esa = ESA(setting)
    #lda = LDA(setting)

    # Preprocess and Load wiki data
    #esa.clean_and_load_data()

    # Create dictionary and corpus
    #esa.build_esa_db()

    # Create TF-IFD vectors for wiki concepts
    # Create the inverted index to map words with concepts
    #esa.create_tf_idf_vectors()
    #p = pstats.Stats('profile.txt')
    #p.sort_stats('time').print_stats(10)

    #esa.prun_inverted_index()

    #esa.load_esa_index()

    # Load Stack data and create TF-IDF vectors
    # Calculate similarities
    #esa.calculate_similarities()
    #esa.calculate_tf_idf_similarities()
    #filtered_users = esa.calculate_local_esa_similarities()
    #filtered_users = esa.calculate_local_tfidf_similarities()
    

    #esa.stack_importer.filtered_users = filtered_users

    #print esa.stack_importer.filtered_users

    #esa.save_index_to_file('../data/ESA/index.txt')
    #esa.save_relatedness_to_file("../data/ESA/relatedness.txt")

    esa.calculate_tfidf_similarities_to_users()
    #esa.calculate_esa_similarities_to_users()


    ###############################################################################
    # Experiments - Calculate statistics on the data
    ###############################################################################
    #esa.initialize_experiments()
    #esa.run_experiment_2_avg(algorithm="tfidf")
    esa.run_experiment_3_avg(algorithm="tfidf")

    #esa.stack_importer.open_stack_db()
    #esa.stack_importer.user_avg_answers()
    #esa.stack_importer.close_stack_db()


    ###############################################################################
    # Running Local LDA
    ###############################################################################
    #lda.calculate_similarities()
    #filtered_users = lda.calculate_local_similarities()
    #lda.stack_importer.filtered_users = filtered_users
    #lda.run_experiment_2_avg()

    # Read dictionary
    """
    connection = sqlite3.connect('../data/ESA/esa.db')
    cursor = connection.cursor()
    query  = 'SELECT term, term_id FROM term_map'
    cursor.execute(query)

    dictionary = []
    for term, term_id in cursor:
        dictionary.append( (term_id, term) )

    with open("dictionary.txt", 'a') as f:
        for term_id, term in dictionary:
            f.write(str(term_id) + " " + term + '\n')
    
    connection = sqlite3.connect('../data/Wiki/enwiki.db')
    cursor = connection.cursor()
    query  = 'SELECT content FROM wiki_articles WHERE id = 1'
    cursor.execute(query)

    with open("content1.txt", 'a') as f:
        for content in cursor:
            f.write(str(content))
    """

    #esa.testing_beer_concept()
    
