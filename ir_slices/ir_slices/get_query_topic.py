from IPython import embed

import pickle
import argparse
import pandas as pd

flatten = lambda l: [item for sublist in l for item in sublist]


base_path = "../../data/"
def get_dialogue_domains_and_intent(df, task, dataset_split):
    if task == "ms_v2":
        json_data = pd.read_json(base_path + "ms_v2/MSDialog-Complete.json")
        json_data_intent = pd.read_json(base_path + "ms_v2/MSDialog-Intent.json")
        json_to_tsv = pd.read_csv(base_path + "ms_v2/"+dataset_split+"_context_query_reply.tsv", \
                                  sep="\t", names=['id', 'c2', 'c3', 'c4', 'c5'])

        def get_domain(ident, json_data):
            json_id = ident.split('-')[1]
            return json_data[int(json_id)]['category']

        def get_intent_last_utterance(ident, json_data):
            json_id = ident.split('-')[1]
            utterance_pos = ident.split('-')[-1]
            if int(json_id) in json_data:
                return json_data[int(json_id)]['utterances'][int(utterance_pos) - 1]['tags']
            else:
                return "NOT_LABELED"

        df_join = df.join(json_to_tsv)
        df_join["category"] = df_join.apply(lambda r, j=json_data, f=get_domain: \
                                                f(r["id"], j), axis=1)
        df_join["intents"] = df_join.apply(lambda r, j=json_data_intent, f=get_intent_last_utterance: \
                                               f(r["id"], j), axis=1)
        return df_join
    elif task == "mantis_10":
        json_data = pd.read_json(base_path + "mantis_10/merged_"+dataset_split+".json")
        json_data_intent = pd.read_json(base_path + "mantis_10/merged_"+dataset_split+"_intents.json")
        json_to_tsv = pd.read_csv(base_path + "mantis_10/data_"+dataset_split+"_easy_lookup.txt",
                                  sep="\t", names=['id'])
        json_to_tsv_only_label_1 = []
        for i, r in json_to_tsv.iterrows():
            if (i + 1) % 11 == 0:
                json_to_tsv_only_label_1.append(r)
        json_to_tsv_df = pd.DataFrame(json_to_tsv_only_label_1).reset_index().drop(['index'], axis=1)
        df_join = df.join(json_to_tsv_df)
        df_join["category"] = df_join.apply(lambda r, j=json_data: \
                                                j[int(r["id"])]["category"], axis=1)

        def get_intent_last_utterance(ident, json_data):
            def isNaN(num):
                return num != num

            if int(ident) in json_data and not isNaN(json_data[int(ident)]["has_intent_labels"]):
                all_labels = [u['intent'] for u in \
                              json_data[int(ident)]['utterances'] if u['actor_type'] != 'agent']
                return " ".join(flatten(all_labels))
            else:
                return "NOT_LABELED"

        df_join["intents"] = df_join.apply(lambda r, j=json_data_intent, f=get_intent_last_utterance: \
                                               f(r["id"], j), axis=1)
        return df_join
    elif task == "mantis_50":
        json_data = pd.read_json(base_path + "mantis_10/merged_"+dataset_split+".json")
        json_data_intent = pd.read_json(base_path + "mantis_10/merged_"+dataset_split+"_intents.json")
        json_to_tsv = pd.read_csv(base_path + "mantis_50/data_"+dataset_split+"_lookup.txt",
                                  sep="\t", names=['id'])
        json_to_tsv_only_label_1 = []
        for i, r in json_to_tsv.iterrows():
            if (i + 1) % 51 == 0:
                json_to_tsv_only_label_1.append(r)
        json_to_tsv_df = pd.DataFrame(json_to_tsv_only_label_1).reset_index().drop(['index'], axis=1)
        df_join = df.join(json_to_tsv_df)

        df_join["category"] = df_join.apply(lambda r, j=json_data: \
                                                j[int(r["id"])]["category"], axis=1)

        def get_intent_last_utterance(ident, json_data):
            def isNaN(num):
                return num != num

            if int(ident) in json_data and not isNaN(json_data[int(ident)]["has_intent_labels"]):
                all_labels = [u['intent'] for u in \
                              json_data[int(ident)]['utterances'] if u['actor_type'] != 'agent']
                return " ".join(flatten(all_labels))
            else:
                return "NOT_LABELED"

        df_join["intents"] = df_join.apply(lambda r, j=json_data_intent, f=get_intent_last_utterance: \
                                               f(r["id"], j), axis=1)
        return df_join

def load_dataset_crr(dataset_path):
    dataset = []
    with open(dataset_path,'r') as f:
        for line in f:
            query = line.split("\t")[1:-1]
            doc = line.split("\t")[-1][0:-2]
            rel = line.split("\t")[0]
            dataset.append((rel, query, doc))
    return dataset

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--dataset_path",
                    type=str,
                    help="dataset path")

parser.add_argument("--task",
                    type=str,
                    help="task")

parser.add_argument("--dataset_split",
                    type=str,
                    help="test, valid or train")


args = parser.parse_args()



if 'ms_v2' in args.dataset_path or 'mantis_10' in args.dataset_path \
        or 'mantis_50' in args.dataset_path:
    dataset = load_dataset_crr(args.dataset_path)
else:
    raise Exception('ms_v2 and mantis_10/50 are the only'
                    ' available datasets with category info for queries.')

df_dataset = pd.DataFrame(dataset, columns = ['label','query','response'])
df_dataset = df_dataset[df_dataset["label"]=='1'].reset_index()

df_with_cat_and_intents = get_dialogue_domains_and_intent(df_dataset, args.task, args.dataset_split)
print(df_with_cat_and_intents.groupby('category').
      agg('count').
      sort_values('query', ascending=False)[0:10])

cat_dict = {}
for idx, row in df_with_cat_and_intents.iterrows():
    cat_dict[' '.join(row['query'])] = row['category']

with open(base_path+args.task+"/cats_"+args.dataset_split+".pickle", 'wb') as f:
    pickle.dump(cat_dict, f)