import numpy as np
import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from snorkel.slicing import SlicingFunction, slicing_function

from IPython import embed

random_sf_count = 0

@slicing_function()
def all_instances(_):
    return True

def random_slice_percentage(_, percentage):
    return random.randint(0,100) <=  (percentage)

#--------------------------#
# Slicing Functions for IR #
#--------------------------#


# Q based
def query_wc_bigger_than(x, l):
    return len(x.query[0].split(" ")) > l

def word_in_query(x, word):
    return word in x.query[0]

def num_turns_bigger_than(x, threshold):
    return len(x.query) > threshold

# Q-D-Y based
def words_match_count_less_than(x, threshold):
    relevant_doc = x.documents[0]
    return len(set(x.query[0].split(" ")).
               intersection(relevant_doc.split(" "))) < threshold

def docs_sim_to_rel_bigger_than(x, threshold, top_k):
    docs = [" ".join(doc.split(" ")[0:min(len(doc.split(" ")), 20)])
            for doc in x.documents]
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_docs = tf_idf_vectorizer.\
        fit_transform(docs)
    cosine_similarities = cosine_similarity(tf_idf_docs[0:1],
                                        tf_idf_docs[1:]).flatten()
    sorted(cosine_similarities, reverse=True)
    top_sim = cosine_similarities[0:min(len(cosine_similarities),top_k)]
    return np.mean(top_sim) > threshold

#--------------------------#
# Function generators      #
#--------------------------#

def make_random_slice_percentage_sf(percentage, count_for_name):
    return SlicingFunction(
        name=f"random_slice_percentage_{percentage}_{count_for_name}",
        f=random_slice_percentage,
        resources=dict(percentage=percentage),
    )

def make_query_wc_bigger_than_sf(l):
    return SlicingFunction(
        name=f"query_wc_bigger_than_{l}",
        f=query_wc_bigger_than,
        resources=dict(l=l),
    )

def make_word_in_query_sf(word):
    return SlicingFunction(
        name=f"word_in_query_{word}",
        f=word_in_query,
        resources=dict(word=word),
    )

def make_words_match_count_less_than_sf(threshold):
    return SlicingFunction(
        name=f"words_match_count_less_than_{threshold}",
        f=words_match_count_less_than,
        resources=dict(threshold=threshold),
    )

def make_num_turns_bigger_than_sf(threshold):
    return SlicingFunction(
        name=f"num_turns_bigger_than_{threshold}",
        f=num_turns_bigger_than,
        resources=dict(threshold=threshold),
    )

def make_docs_sim_to_rel_bigger_than_sf(threshold, top_k):
    threshold_str = str(threshold).replace(".", "dot")
    return SlicingFunction(
        name=f"docs_sim_to_rel_bigger_than_{threshold_str}_top_{top_k}",
        f=docs_sim_to_rel_bigger_than,
        resources=dict(threshold=threshold, top_k=top_k),
    )

slicing_functions = {
    "quora" : [
        # all_instances,
        make_docs_sim_to_rel_bigger_than_sf(0.3, 3),
        make_docs_sim_to_rel_bigger_than_sf(0.4, 3),
        make_docs_sim_to_rel_bigger_than_sf(0.5, 3),
        make_query_wc_bigger_than_sf(10),
        make_query_wc_bigger_than_sf(15),
        make_word_in_query_sf("who"),
        make_word_in_query_sf("what"),
        # make_word_in_query_sf("where"), # only 2% of dev data
        # make_word_in_query_sf("when"), # only 1% of dev data
        make_word_in_query_sf("why"),
        make_word_in_query_sf("how"),
        # make_words_match_count_less_than_sf(3), # only 2% of dev data
        make_words_match_count_less_than_sf(4),
        make_words_match_count_less_than_sf(5)
    ],
    "l4": [
        # all_instances,
        make_docs_sim_to_rel_bigger_than_sf(0.1, 2),
        make_query_wc_bigger_than_sf(15),
        make_query_wc_bigger_than_sf(20),
        make_query_wc_bigger_than_sf(25),
        # make_word_in_query_sf("who"), # only 2% of dev data
        make_word_in_query_sf("what"),
        # make_word_in_query_sf("where"), # only 2% of dev data
        make_word_in_query_sf("when"),
        # make_word_in_query_sf("why"),  # <1% of dev data
        # make_word_in_query_sf("how"), every L4 instance is a 'how' question
        make_words_match_count_less_than_sf(2),
        make_words_match_count_less_than_sf(3),
        make_words_match_count_less_than_sf(4)
    ],
    "mantis_10":[
        # all_instances,
        make_docs_sim_to_rel_bigger_than_sf(0.10, 2),
        # make_docs_sim_to_rel_bigger_than_sf(0.4, 3),
        # make_docs_sim_to_rel_bigger_than_sf(0.5, 3),
        make_query_wc_bigger_than_sf(150),
        make_query_wc_bigger_than_sf(250),
        make_query_wc_bigger_than_sf(300),
        make_query_wc_bigger_than_sf(400),
        make_word_in_query_sf("who"),
        make_word_in_query_sf("what"),
        make_word_in_query_sf("where"),
        make_word_in_query_sf("when"),
        make_word_in_query_sf("why"),
        make_word_in_query_sf("how"),
        make_words_match_count_less_than_sf(2),
        make_words_match_count_less_than_sf(3),
        make_words_match_count_less_than_sf(4),
        make_num_turns_bigger_than_sf(4),
        make_num_turns_bigger_than_sf(6),
        make_num_turns_bigger_than_sf(8)],
    "ms_v2": [
        # all_instances,
        make_docs_sim_to_rel_bigger_than_sf(0.15, 2),
        make_docs_sim_to_rel_bigger_than_sf(0.2, 2),
        make_query_wc_bigger_than_sf(150),
        make_query_wc_bigger_than_sf(250),
        make_query_wc_bigger_than_sf(300),
        make_query_wc_bigger_than_sf(400),
        make_word_in_query_sf("who"),
        make_word_in_query_sf("what"),
        make_word_in_query_sf("where"),
        make_word_in_query_sf("when"),
        make_word_in_query_sf("why"),
        make_word_in_query_sf("how"),
        make_words_match_count_less_than_sf(2),
        make_words_match_count_less_than_sf(3),
        make_words_match_count_less_than_sf(4),
        make_num_turns_bigger_than_sf(4),
        make_num_turns_bigger_than_sf(6),
        make_num_turns_bigger_than_sf(8)],
    "ms_marco_adhoc":
        [
        # all_instances,
         make_docs_sim_to_rel_bigger_than_sf(0.15, 2),
         make_docs_sim_to_rel_bigger_than_sf(0.20, 2),
         make_query_wc_bigger_than_sf(5),
         make_query_wc_bigger_than_sf(8),
         make_word_in_query_sf("who"),
         make_word_in_query_sf("what"),
         make_word_in_query_sf("where"),
         make_word_in_query_sf("when"),
         # make_word_in_query_sf("why"), # less than 1%
         make_word_in_query_sf("how"),
         make_words_match_count_less_than_sf(2),
         make_words_match_count_less_than_sf(3),
         make_words_match_count_less_than_sf(4)],
    "udc":
        [
        # all_instances,
        make_docs_sim_to_rel_bigger_than_sf(0.10, 2),
        make_docs_sim_to_rel_bigger_than_sf(0.15, 2),
        make_query_wc_bigger_than_sf(10),
        make_query_wc_bigger_than_sf(20),
        make_query_wc_bigger_than_sf(50),
        # make_word_in_query_sf("who"), # less than 2%
        make_word_in_query_sf("what"),
        make_word_in_query_sf("where"),
        make_word_in_query_sf("when"),
        make_word_in_query_sf("why"),
        make_word_in_query_sf("how"),
        make_words_match_count_less_than_sf(1),
        make_words_match_count_less_than_sf(2),
        make_words_match_count_less_than_sf(3),
        make_num_turns_bigger_than_sf(10),
        make_num_turns_bigger_than_sf(15),
        make_num_turns_bigger_than_sf(17)]
}


random_slicing_functions = {
    "quora" : [
        # all_instances,
        make_random_slice_percentage_sf(50,1),
        make_random_slice_percentage_sf(50,2),
        make_random_slice_percentage_sf(50,3),
        make_random_slice_percentage_sf(50,4),
        make_random_slice_percentage_sf(50,5),
        make_random_slice_percentage_sf(50,6),
        make_random_slice_percentage_sf(50,7),
        make_random_slice_percentage_sf(50,8),
        make_random_slice_percentage_sf(50,9),
        make_random_slice_percentage_sf(50,10)
    ],
    "l4": [
        # all_instances,
        make_random_slice_percentage_sf(50,1),
        make_random_slice_percentage_sf(50,2),
        make_random_slice_percentage_sf(50,3),
        make_random_slice_percentage_sf(50,4),
        make_random_slice_percentage_sf(50,5),
        make_random_slice_percentage_sf(50,6),
        make_random_slice_percentage_sf(50,7),
        make_random_slice_percentage_sf(50,8),
        make_random_slice_percentage_sf(50,9),
        make_random_slice_percentage_sf(50,10)
    ],
    "mantis_10":[
        # all_instances,
        make_random_slice_percentage_sf(50,1),
        make_random_slice_percentage_sf(50,2),
        make_random_slice_percentage_sf(50,3),
        make_random_slice_percentage_sf(50,4),
        make_random_slice_percentage_sf(50,5),
        make_random_slice_percentage_sf(50,6),
        make_random_slice_percentage_sf(50,7),
        make_random_slice_percentage_sf(50,8),
        make_random_slice_percentage_sf(50,9),
        make_random_slice_percentage_sf(50,10)],
    "ms_v2": [
        # all_instances,
        make_random_slice_percentage_sf(50,1),
        make_random_slice_percentage_sf(50,2),
        make_random_slice_percentage_sf(50,3),
        make_random_slice_percentage_sf(50,4),
        make_random_slice_percentage_sf(50,5),
        make_random_slice_percentage_sf(50,6),
        make_random_slice_percentage_sf(50,7),
        make_random_slice_percentage_sf(50,8),
        make_random_slice_percentage_sf(50,9),
        make_random_slice_percentage_sf(50,10)],
    "ms_marco_adhoc":
        [
        # all_instances,
        make_random_slice_percentage_sf(50, 1),
        make_random_slice_percentage_sf(50, 2),
        make_random_slice_percentage_sf(50, 3),
        make_random_slice_percentage_sf(50, 4),
        make_random_slice_percentage_sf(50, 5),
        make_random_slice_percentage_sf(50, 6),
        make_random_slice_percentage_sf(50, 7),
        make_random_slice_percentage_sf(50, 8),
        make_random_slice_percentage_sf(50, 9),
        make_random_slice_percentage_sf(50, 10)],
    "udc":
        [
        make_random_slice_percentage_sf(50, 1),
        make_random_slice_percentage_sf(50, 2),
        make_random_slice_percentage_sf(50, 3),
        make_random_slice_percentage_sf(50, 4),
        make_random_slice_percentage_sf(50, 5),
        make_random_slice_percentage_sf(50, 6),
        make_random_slice_percentage_sf(50, 7),
        make_random_slice_percentage_sf(50, 8),
        make_random_slice_percentage_sf(50, 9),
        make_random_slice_percentage_sf(50, 10)]
}
