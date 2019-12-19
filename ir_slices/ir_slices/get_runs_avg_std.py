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
        print(f_run_path)
        with open(f_run_path, 'r') as f_run, open(f_config_path, 'r') as f_config:
            res = json.load(f_run)
            config = json.load(f_config)
            map = res['result']['map']
            model = config['args']['model_type']
            all_res.append([model, map, f_run_path.split('data')[-1]])
    out_df = pd.DataFrame(all_res, columns=['model', 'map', 'run'])
    arg_max = out_df.sort_values('map', ascending=False).drop_duplicates(['model'])

    agg_df = out_df.groupby("model").\
        agg(['mean', 'std', 'count', 'max']).\
        reset_index().round(3)
    agg_df['dataset'] = args.task_name
    agg_df.columns = ['model','Avg. MAP', 'std', 'count', 'max', 'dataset']
    agg_df = agg_df.merge(arg_max[['model', 'run']], on='model')
    agg_df['model'] = agg_df.apply(lambda r: r['model'].upper(), axis=1)
    agg_df['model'][agg_df['model']!= 'BERT'] = agg_df[agg_df['model']!= 'BERT'].apply(lambda r: "\\bertsliceaware" if r['model'] == 'BERT-SLICE-AWARE' else "\\bertsliceawarerandom", axis=1)
    agg_df['Avg. MAP (std)'] = agg_df.apply(lambda r: str(r['Avg. MAP']) + " (." + str(r['std']).split(".")[1] + ")", axis=1)
    agg_df[['dataset', 'model', 'Avg. MAP (std)', 'max', 'count', 'run']].sort_values("max").\
        to_csv(args.output_folder+args.task_name+"_agg_res.txt",
                  sep='\t')

if __name__ == "__main__":
    main()
