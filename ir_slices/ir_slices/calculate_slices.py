from snorkel.slicing import SFApplier
from IPython import embed

from ir_slices.data_processors import processors
from ir_slices.slice_functions import slicing_functions

import numpy as np
import scipy.stats

import pandas as pd
import argparse

pd.set_option('display.max_columns', None)

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--predictions_file", default="", type=str,
                        help="The file with the predictions for each <query,document> pair")
    parser.add_argument("--eval_metrics_files", default="", type=str,
                        help="The files with the metrics (e.g. AP, nDCG) for each query, separated by ;")
    parser.add_argument("--eval_models", default="", type=str,
                        help="The names of the models separated by ;")

    args = parser.parse_args()

    eval_metrics = ['AP']
    dfs = []
    for model_eval_file, model_name in zip(args.eval_metrics_files.split(";"),
                                           args.eval_models.split(";")):
        df_eval = pd.read_csv(model_eval_file, names=eval_metrics)

        processor = processors[args.task_name]()
        examples = processor.get_dev_examples(args.data_dir)

        per_slice_results = []
        for slice_function in slicing_functions[args.task_name]:
            slice = [slice_function(example) for example in examples]
            df_eval[slice_function.name] = slice
            for metric in eval_metrics:
                per_slice_results.append([args.task_name, model_name, slice_function.name, metric,\
                                          df_eval[df_eval[slice_function.name]][metric].mean(),
                                          confidence_interval(df_eval[df_eval[slice_function.name]][metric]),
                                          df_eval[df_eval[slice_function.name]].shape[0]/df_eval.shape[0],
                                          df_eval[df_eval[slice_function.name]].shape[0]])
        per_slice_results = pd.DataFrame(per_slice_results,
                                        columns=["task", "model", "slice", "metric",
                                                 "value", "ci",
                                                 "%", "N"])
        dfs.append(per_slice_results)
    all_dfs = pd.concat(dfs)
    print(all_dfs.sort_values("value"))

if __name__ == "__main__":
    main()
