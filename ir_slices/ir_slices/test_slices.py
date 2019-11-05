from snorkel.slicing import SFApplier
from IPython import embed

from ir_slices.data_processors import processors
from ir_slices.slice_functions import slicing_functions, random_slicing_functions

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

    args = parser.parse_args()

    processor = processors[args.task_name]()
    examples = processor.get_dev_examples(args.data_dir)

    for slice_function in random_slicing_functions[args.task_name]:
    # for slice_function in slicing_functions[args.task_name]:
        slice = [slice_function(example) for example in examples]
        print(slice_function.name)
        print(sum(slice))
        print(sum(slice)/len(slice))

if __name__ == "__main__":
    main()
