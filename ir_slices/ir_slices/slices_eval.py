from snorkel.slicing import SFApplier
from functools import reduce
from IPython import embed
from tqdm import tqdm

import os
import pickle

from ir_slices.data_processors import processors
from ir_slices.slice_functions import slicing_functions, all_instances

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
import scipy.stats

import pandas as pd
import argparse

import torch

pd.set_option('display.max_columns', None)

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def unpack_x_per_doc(X, examples):
    unpacked = []
    for x, example in zip(X, examples):
        for _ in example.documents:
            unpacked.append(x)
    return unpacked

def unpack_rel_per_doc(examples):
    rel = []
    for example in examples:
        for l in example.labels:
            rel.append(l)
    return rel

def unpack_qid_per_doc(examples):
    qids = {}
    i=0
    queries = []
    for example in examples:
        for _ in example.documents:
            query = " ".join(example.query)
            if query not in qids:
                qids[query] = i
                i+=1
            queries.append(qids[query])
    return queries

def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--eval_metrics_files", default="", type=str,
                        help="The files with the metrics (e.g. AP, nDCG) for each query, separated by ';'")
    parser.add_argument("--eval_models", default="", type=str,
                        help="The names of the models separated by ';'")
    parser.add_argument("--representations_files", default="", type=str,
                        help="The files with representations (for each {Q,D} pair) of each model separated by ';'")
    parser.add_argument("--slice_scores_file", default="", type=str,
                        help="pickle file with per slice scores.")
    parser.add_argument("--output_folder", default="", type=str,
                        help="")

    args = parser.parse_args()

    if args.representations_files != "" :
        rep_files = args.representations_files.split(";")
    else:
        rep_files = [""] * len(args.eval_metrics_files.split(";"))

    eval_metrics = ['AP']
    eval_dfs = []
    rep_dfs = []
    for model_eval_file, model_name, rep_file in zip(args.eval_metrics_files.split(";"),
                                                     args.eval_models.split(";"),
                                                     rep_files):
        df_eval = pd.read_csv(model_eval_file, names=eval_metrics)
        processor = processors[args.task_name]()
        examples = processor.get_dev_examples(args.data_dir)

        if rep_file != "":
            df_rep = pd.DataFrame(torch.load(rep_file, map_location='cpu').numpy())
            num_instances_rep = df_rep.shape[0]
            df_tsne = TSNE(n_components=2, verbose=True).fit_transform(df_rep)
            df_rep['TSNE_0'] = df_tsne[:, 0]
            df_rep['TSNE_1'] = df_tsne[:, 1]

            df_pca = PCA(n_components=2).fit_transform(df_rep)
            df_rep['PC_0'] = df_pca[:, 0]
            df_rep['PC_1'] = df_pca[:, 1]

            df_rep['model'] = model_name
            df_rep[eval_metrics[0]] = unpack_x_per_doc(df_eval[eval_metrics[0]].values,
                                                       examples)[:num_instances_rep]
            df_rep['relevant'] = unpack_rel_per_doc(examples)[:num_instances_rep]
            df_rep['q_id'] = unpack_qid_per_doc(examples)[:num_instances_rep]

        per_slice_results = []
        for slice_function in slicing_functions[args.task_name] + [all_instances]:
            # This only accepts a single fine tuned bert slicing function for each task.
            if "fine_tuned_bert_pred" in slice_function.name:
                if os.path.isfile(args.data_dir+"/cached_ft_predictions_valid.pickle"):
                    with open(args.data_dir+"/cached_ft_predictions_valid.pickle", "rb") as f:
                        print("loaded pickle for fine tuned predictions")
                        slice = pickle.load(f)
                else:
                    slice = [slice_function(example) for example in tqdm(examples)]
                    with open(args.data_dir+"/cached_ft_predictions_valid.pickle", "wb") as f:
                        pickle.dump(slice, f)
            else:
                slice = [slice_function(example) for example in tqdm(examples)]
            df_eval[slice_function.name] = slice
            if rep_file != "":
                df_rep[slice_function.name] = unpack_x_per_doc(slice,
                                                               examples)[:num_instances_rep]
            for metric in eval_metrics:
                per_slice_results.append([args.task_name, model_name, slice_function.name, metric,\
                                          df_eval[df_eval[slice_function.name]][metric].mean(),
                                          df_eval[df_eval[slice_function.name]][metric],
                                          np.std(df_eval[df_eval[slice_function.name]][metric]),
                                          confidence_interval(df_eval[df_eval[slice_function.name]][metric]),
                                          df_eval[df_eval[slice_function.name]].shape[0]/df_eval.shape[0],
                                          df_eval[df_eval[slice_function.name]].shape[0]])
        per_slice_results = pd.DataFrame(per_slice_results,
                                        columns=["task", "model", "slice", "metric",
                                                 "value", "all_values", "std" ,"ci",
                                                 "%", "N"])
        eval_dfs.append(per_slice_results)
        if rep_file != "":
            rep_dfs.append(df_rep)

    # saving df with representations
    all_rep_dfs = pd.concat(rep_dfs)
    all_rep_dfs.to_csv(args.output_folder+"rep_res_"+args.task_name)

    # saving df with sliced evaluation
    all_eval_dfs = pd.concat(eval_dfs)
    all_eval_dfs.sort_values(["model", "value"]).to_csv(args.output_folder+"res_"+args.task_name)

    # calculating delta between baseline and competitor approach
    slice_membership_s = []
    with open(args.slice_scores_file, 'rb') as f:
        slice_membership_scores = pickle.load(f)
        for k in slice_membership_scores.keys():
            if 'ind' in k and 'base' not in k and 'f1' in k :
                slice_membership_s.append([k.split(":")[1].split("_ind")[0],
                                             slice_membership_scores[k]])
    slice_membership_s_df = pd.DataFrame(slice_membership_s, columns=['slice', 'slice_membership_f1'])

    df_final = reduce(lambda left, right: pd.merge(left, right, on='slice'), eval_dfs)
    df_final = df_final.merge(slice_membership_s_df, on='slice')
    df_final['delta'] = df_final['value_y']-df_final['value_x']
    # delta from the random to the non-random slice-aware model
    df_final['delta_to_random'] = df_final['value_y'] - df_final['value']
    # delta from the random slicing functions to baseline
    df_final['delta_random_to_baseline'] = df_final['value'] - df_final['value_x']
    df_final['p_value'] = df_final.apply(lambda x,f=scipy.stats.ttest_ind:
                                         f(x['all_values_y'], x['all_values_x'])[1], axis=1)
    df_final['p_value_random_sf'] = df_final.apply(lambda x, f=scipy.stats.ttest_ind:
                                         f(x['all_values'], x['all_values_x'])[1], axis=1)
    df_final['p_value<0.05'] = df_final['p_value']<0.05
    df_final['p_value<0.01'] = df_final['p_value']<0.01
    df_final['random_p_value<0.05'] = df_final['p_value_random_sf'] < 0.05
    df_final['random_p_value<0.01'] = df_final['p_value_random_sf'] < 0.01
    df_final.to_csv(args.output_folder+"delta_res_" + args.task_name)

    # generating table 1 of article
    table_1 = all_eval_dfs[all_eval_dfs['slice']=='all_instances']\
        [['task', 'model', 'value', 'ci']]
    table_1.columns = ['task', 'model', 'value', 'ci']

    pvalue_all_instances = df_final[df_final['slice']=='all_instances'][['model_y','p_value<0.05', 'random_p_value<0.05']]
    pvalue_all_instances.columns = ['model', 'p_value<0.05', 'random_sf_p_value<0.05']
    table_1 = table_1.\
        merge(pvalue_all_instances, on=['model'], how='outer'). \
        replace(np.nan, '-', regex=True)

    lifts = df_final.groupby('model_y')['delta'].agg(['mean', 'max']).reset_index()
    lifts.columns = ['model', 'Avg. slice lift', 'Max. slice lift']

    lifts_random = df_final.groupby('model')['delta_random_to_baseline'].agg(['mean', 'max']).reset_index()
    lifts_random.columns = ['model', 'Avg. slice lift', 'Max. slice lift']

    lifts_both = pd.concat([lifts, lifts_random])
    table_1 = table_1.\
        merge(lifts_both, on=['model'], how='outer'). \
        replace(np.nan, '-', regex=True)
    print(table_1)
    table_1.to_csv(args.output_folder+"table_1_" + args.task_name, sep='\t')

if __name__ == "__main__":
    main()
