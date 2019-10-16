
def slice_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        func_name = func.__name__ + "_"+ str(*args[1:])
        print("Slice \'"+func_name+"\' includes: " +
              str(round(sum(result)/len(result),2)*100)+" % of data.")
        # if sum(result) > 1:
        #     i=0
        #     while not result[i]:
        #         i+=1
        #     if(i<len(result)):
        #         print("Slice \'"+func_name+"\' example: " +
        #               str(result[i]) + ", instance: " + str(args[0][i]))

        return result
    return wrapper

@slice_decorator
def query_wc_bigger_than(instances, l=10):
    return [len(instance.query[0].split(" ")) > l for instance in instances]

@slice_decorator
def word_in_query(instances, word):
    return [word in instance.query[0] for instance in instances]

@slice_decorator
def words_match_count_less_than(instances, threshold=2):
    return [len(set(instance.query[0].split(" ")).intersection(instance.documents[0].split(" ")))
            < threshold for instance in instances]

slicing_functions = {
    "l4":
        [("query_wc_bigger_than_5", lambda x: query_wc_bigger_than(x, 10)),
         ("query_wc_bigger_than_15", lambda x: query_wc_bigger_than(x, 15)),
         ("word_in_query_what", lambda x: word_in_query(x, "what")),
         # ("word_in_query_how", lambda x: word_in_query(x, "how")), # all queries in L4 start with how
         ("word_in_query_who", lambda x: word_in_query(x, "who")),
         ("word_in_query_why", lambda x: word_in_query(x, "why")),
         ("word_in_query_which", lambda x: word_in_query(x, "which")),
         ("word_in_query_where", lambda x: word_in_query(x, "where")),
         ("words_match_count_less_than_2", lambda x: words_match_count_less_than(x, 2)),
         ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 3)),
         ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 4))],
    "quora" :
        [("query_wc_bigger_than_5", lambda x : query_wc_bigger_than(x, 10)),
        ("query_wc_bigger_than_15", lambda x : query_wc_bigger_than(x, 15)),
        ("word_in_query_what", lambda x : word_in_query(x, "what")),
        ("word_in_query_how", lambda x : word_in_query(x, "how")),
        ("word_in_query_who", lambda x : word_in_query(x, "who")),
        ("word_in_query_why", lambda x : word_in_query(x, "why")),
        ("word_in_query_which", lambda x : word_in_query(x, "which")),
        ("word_in_query_where", lambda x : word_in_query(x, "where")),
         ("words_match_count_less_than_2", lambda x: words_match_count_less_than(x, 3)),
         ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 5)),
         ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 7))],
    "mantis_10":
        [("query_wc_bigger_than_512", lambda x: query_wc_bigger_than(x, 512)),
         ("word_in_query_what", lambda x: word_in_query(x, "what")),
         ("word_in_query_how", lambda x: word_in_query(x, "how")),
         ("word_in_query_who", lambda x: word_in_query(x, "who")),
         ("word_in_query_why", lambda x: word_in_query(x, "why")),
         ("word_in_query_which", lambda x: word_in_query(x, "which")),
         ("word_in_query_where", lambda x: word_in_query(x, "where")),
         ("words_match_count_less_than_2", lambda x: words_match_count_less_than(x, 3)),
         ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 5)),
         ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 7))],
    "ms_marco_adhoc":
        [("word_in_query_what", lambda x: word_in_query(x, "what")),
         ("word_in_query_how", lambda x: word_in_query(x, "how")),
         ("word_in_query_who", lambda x: word_in_query(x, "who")),
         ("word_in_query_why", lambda x: word_in_query(x, "why")),
         ("word_in_query_which", lambda x: word_in_query(x, "which")),
         ("word_in_query_where", lambda x: word_in_query(x, "where")),
         ("query_wc_bigger_than_5", lambda x : query_wc_bigger_than(x, 10)),
         ("query_wc_bigger_than_15", lambda x : query_wc_bigger_than(x, 15)),
         ("words_match_count_less_than_2", lambda x: words_match_count_less_than(x, 1)),
         ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 2)),
         ("words_match_count_less_than_3", lambda x: words_match_count_less_than(x, 3))]
}

