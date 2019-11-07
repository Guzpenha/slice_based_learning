import numpy as np
import random
import torch

from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)

from transformers import (BertTokenizer,
                          BertForSequenceClassification)
from transformers import glue_convert_examples_to_features \
    as convert_examples_to_features
from transformers.data.processors.utils import InputExample

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.special import softmax

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

def fine_tuned_bert_pred_diff_smaller_than(x, threshold,
                                           model, tokenizer):
    instances = []
    for doc, label in zip(x.documents,
                          x.labels):
        instances.append(
            InputExample(guid='Q',
                         text_a=" ".join(x.query),
                         text_b=doc,
                         label=label))
    features = convert_examples_to_features(instances,
                                            tokenizer,
                                            label_list=["0", "1"],
                                            max_length=512,
                                            output_mode='ranking',
                                            pad_on_left=False,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=0
                                            )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=len(instances))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            _, logits = outputs[:2]
            preds = logits.detach().cpu().numpy()
            preds = softmax(preds, axis=1)
            preds = preds[:, 1]

    # relevant document is always the first one
    diff = preds[0] - max(preds[1:])

    return diff < threshold

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

def make_fine_tuned_bert_pred_diff_smaller_than_sf(finetunning_task, threshold,
                                                   model, tokenizer):
    threshold_str = str(threshold).replace(".", "dot")
    return SlicingFunction(
        name=f"fine_tuned_bert_pred_diff_smaller_than{threshold_str}_finetuned_on_{finetunning_task}",
        f=fine_tuned_bert_pred_diff_smaller_than,
        resources=dict(model=model, tokenizer=tokenizer,
                       threshold=threshold),
    )

fine_tuned_bert_paths = {
    # 'quora': '../../data/quora_output/bert/',
    # 'l4': '../../data/l4_output/bert/',
    # 'mantis_10': '../../data/mantis_10_output/bert/',
    'ms_v2': '../../data/ms_v2_output/bert/',
    # 'ms_marco_adhoc': '../../data/ms_marco_adhoc_output/bert/',
    # 'udc': '../../data/udc_output/bert/'
}
fine_tuned_models = {}

for finetuning_task in fine_tuned_bert_paths.keys():
    model_path = fine_tuned_bert_paths[finetuning_task]
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(model_path)
    fine_tuned_models[finetuning_task] = (model, tokenizer)

slicing_functions = {
    "quora" : [
        # all_instances,
        make_fine_tuned_bert_pred_diff_smaller_than_sf('ms_v2', 0.2,
                                                       fine_tuned_models['ms_v2'][0],
                                                       fine_tuned_models['ms_v2'][1]),
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
        make_fine_tuned_bert_pred_diff_smaller_than_sf('ms_v2', 0.1,
                                                       fine_tuned_models['ms_v2'][0],
                                                       fine_tuned_models['ms_v2'][1]),
        make_docs_sim_to_rel_bigger_than_sf(0.2, 2),
        make_query_wc_bigger_than_sf(150),
        make_word_in_query_sf("who"),
        make_word_in_query_sf("what"),
        make_word_in_query_sf("where"),
        make_word_in_query_sf("when"),
        make_word_in_query_sf("why"),
        make_word_in_query_sf("how"),
        make_words_match_count_less_than_sf(4),
        make_num_turns_bigger_than_sf(6)],
    "ms_marco_adhoc":[
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
    "udc":[
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
