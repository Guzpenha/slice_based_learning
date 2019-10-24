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

# Q-D based


# Q-D-Y based
def words_match_count_less_than(x, threshold):
    return len(set(x.query[0].split(" ")).
               intersection(x.documents[0].split(" "))) < threshold

#--------------------------#
# Function generators      #
#--------------------------#

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

def make_random_slice_percentage_sf(percentage):
    return SlicingFunction(
        name=f"random_slice_percentage_{percentage}",
        f=random_slice_percentage,
        resources=dict(percentage=percentage),
    )

slicing_functions = {
    "quora" : [
        all_instances,
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
    ]
    # "l4":
    #     [("all_instances", lambda x: all_instances(x)),
    #      ("query_wc_bigger_than_5", lambda x: make_query_wc_bigger_than_sf(10)),
    #      ("query_wc_bigger_than_15", lambda x: query_wc_bigger_than(x, 15)),
    #      ("word_in_query_what", lambda x: word_in_query(x, "what")),
    #      # ("word_in_query_how", lambda x: word_in_query(x, "how")), # all queries in L4 start with how
    #      ("word_in_query_who", lambda x: word_in_query(x, "who")),
    #      ("word_in_query_why", lambda x: word_in_query(x, "why")),
    #      ("word_in_query_which", lambda x: word_in_query(x, "which")),
    #      ("word_in_query_where", lambda x: word_in_query(x, "where")),
    #      ("words_match_count_less_than_2", lambda x: words_match_count_less_than(x, 2)),
    #      ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 3)),
    #      ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 4))],
    # "quora" :
    #     [("all_instances", lambda x: all_instances(x)),
    #     ("query_wc_bigger_than_5", lambda x : make_query_wc_bigger_than_sf(x, l=10)),
    #     ("query_wc_bigger_than_15", lambda x : query_wc_bigger_than(x, 15)),
    #     ("word_in_query_what", lambda x : word_in_query(x, "what")),
    #     ("word_in_query_how", lambda x : word_in_query(x, "how")),
    #     ("word_in_query_who", lambda x : word_in_query(x, "who")),
    #     ("word_in_query_why", lambda x : word_in_query(x, "why")),
    #     ("word_in_query_which", lambda x : word_in_query(x, "which")),
    #     ("word_in_query_where", lambda x : word_in_query(x, "where")),
    #      ("words_match_count_less_than_2", lambda x: words_match_count_less_than(x, 3)),
    #      ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 5)),
    #      ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 7))],
    # "mantis_10":
    #     [("all_instances", lambda x: all_instances(x)),
    #      ("query_wc_bigger_than_512", lambda x: query_wc_bigger_than(x, 512)),
    #      ("word_in_query_what", lambda x: word_in_query(x, "what")),
    #      ("word_in_query_how", lambda x: word_in_query(x, "how")),
    #      ("word_in_query_who", lambda x: word_in_query(x, "who")),
    #      ("word_in_query_why", lambda x: word_in_query(x, "why")),
    #      ("word_in_query_which", lambda x: word_in_query(x, "which")),
    #      ("word_in_query_where", lambda x: word_in_query(x, "where")),
    #      ("words_match_count_less_than_2", lambda x: words_match_count_less_than(x, 3)),
    #      ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 5)),
    #      ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 7))],
    # "ms_marco_adhoc":
    #     [("all_instances", lambda x: all_instances(x)),
    #      ("word_in_query_what", lambda x: word_in_query(x, "what")),
    #      ("word_in_query_how", lambda x: word_in_query(x, "how")),
    #      ("word_in_query_who", lambda x: word_in_query(x, "who")),
    #      ("word_in_query_why", lambda x: word_in_query(x, "why")),
    #      ("word_in_query_which", lambda x: word_in_query(x, "which")),
    #      ("word_in_query_where", lambda x: word_in_query(x, "where")),
    #      ("query_wc_bigger_than_5", lambda x : query_wc_bigger_than(x, 10)),
    #      ("query_wc_bigger_than_15", lambda x : query_wc_bigger_than(x, 15)),
    #      ("words_match_count_less_than_2", lambda x: words_match_count_less_than(x, 1)),
    #      ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 2)),
    #      ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 3))]
}


