from IPython import embed
import pandas as pd

import json
import argparse


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task (e.g. ms_v2).")
    parser.add_argument("--runs", default="", type=str, required=True,
                        help="The runs separated by ';'")
    parser.add_argument("--experiments_folder",type=str,
                        default="/tudelft.net/staff-umbrella/conversationalsearch/slice_based_learning/data/")
    parser.add_argument("--output_folder", type=str,
                        default="/tudelft.net/staff-umbrella/conversationalsearch/slice_based_learning/tmp/")

    args = parser.parse_args()

    runs = args.runs.split(";")
    run_files = [(args.experiments_folder+args.task_name+"_output/"+run+"/run.json",
                  args.experiments_folder+args.task_name+"_output/"+run+"/config.json",)
                 for run in runs]
    all_res = []
    for f_run_path, f_config_path in run_files:
        with open(f_run_path, 'r') as f_run, open(f_config_path, 'r') as f_config:
            res = json.load(f_run)
            config = json.load(f_config)
            map = res['result']['map']
            model = config['args']['model_type']
            all_res.append([model, map])
    out_df = pd.DataFrame(all_res, columns=['model', 'map'])
    agg_df = out_df.groupby("model").agg(['mean', 'std', 'count', 'max'])
    agg_df['model'] = args.task_name
    agg_df.to_csv(args.output_folder+args.task_name+"_agg_res.txt",
                  sep='\t', index=False)

if __name__ == "__main__":
    main()
