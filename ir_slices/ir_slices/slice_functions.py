from snorkel.slicing import SlicingFunction, slicing_function
import random

@slicing_function()
def all_instances(_):
    return True

def random_slice_percentage(_, percentage):
    return random.randint(1,101) <  (percentage)

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

# Q-D based


# Q-D-Y based
def words_match_count_less_than(x, threshold):
    relevant_doc = x.documents[0]
    return len(set(x.query[0].split(" ")).
               intersection(relevant_doc.split(" "))) < threshold

#--------------------------#
# Function generators      #
#--------------------------#

def make_random_slice_percentage_sf(percentage):
    return SlicingFunction(
        name=f"random_slice_percentage_{percentage}",
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

slicing_functions = {
    "quora" : [
        # all_instances,
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
    "quora_random" : [
        make_random_slice_percentage_sf(50),
        make_random_slice_percentage_sf(90),
        make_random_slice_percentage_sf(30)
    ],
    "l4": [
        # all_instances,
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


