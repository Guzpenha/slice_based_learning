import argparse

from slice_calculation.data_processors import processors
from slice_calculation.ir_slices import slicing_functions

def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    # parser.add_argument("--predictions_file", default=None, type=str, required=True,
    #                     help="The file with the predictions for each <query,document> pair")
    # parser.add_argument("--eval_metrics_file", default=None, type=str, required=True,
    #                     help="The file with the metrics (e.g. AP, nDCG) for each query")
    # parser.add_argument("--output_file", default=None, type=str, required=True,
    #                     help="Output file to save slices.")
    args = parser.parse_args()

    processor = processors[args.task_name]()
    # examples = processor.get_train_examples(args.data_dir)
    examples = processor.get_dev_examples(args.data_dir)

    for f_name, slice_function in slicing_functions[args.task_name]:
        slice = slice_function(examples)

if __name__ == "__main__":
    main()
