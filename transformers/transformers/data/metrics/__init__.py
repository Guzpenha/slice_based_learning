# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    import random
    import numpy as np
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def _to_list(x):
        if isinstance(x, list):
            return x
        return [x]

    def ap(y_true, y_pred, rel_threshold=0):
        s = 0.
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = [a for a in zip(y_true, y_pred)]
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        ipos = 0
        for j, (g, p) in enumerate(c):
            if g > rel_threshold:
                ipos += 1.
                s += ipos / (j + 1.)
        if ipos == 0:
            s = 0.
        else:
            s /= ipos
        return s


    def ndcg(k=10):
        def top_k(y_true, y_pred, rel_threshold=0.):
            if k <= 0.:
                return 0.
            s = 0.
            y_true = _to_list(np.squeeze(y_true).tolist())
            y_pred = _to_list(np.squeeze(y_pred).tolist())
            c = [v for v in zip(y_true, y_pred)]
            random.shuffle(c)
            c_g = sorted(c, key=lambda x: x[0], reverse=True)
            c_p = sorted(c, key=lambda x: x[1], reverse=True)
            idcg = 0.
            ndcg = 0.
            for i, (g, p) in enumerate(c_g):
                if i >= k:
                    break
                if g > rel_threshold:
                    idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            for i, (g, p) in enumerate(c_p):
                if i >= k:
                    break
                if g > rel_threshold:
                    ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            if idcg == 0.:
                return 0.
            else:
                return ndcg / idcg

        return top_k


    def ndcg_at_10(preds, labels):
        ndcgs = []
        query_preds = []
        query_labels = []
        i = 0
        ndcg_at_10 = ndcg(k=10)
        for l, p in zip(labels, preds):
            if (l == 1 and i != 0):
                ndcgs.append(ndcg_at_10(query_labels, query_preds))
                query_preds = []
                query_labels = []

            query_preds.append(p)
            query_labels.append(l)

            i += 1

            if (i == len(preds)):
                ndcgs.append(ndcg_at_10(query_labels, query_preds))
        return sum(ndcgs) / float(len(ndcgs))

    def mean_average_precision(preds, labels):
        aps = []
        query_preds = []
        query_labels = []
        i = 0
        for l, p in zip(labels, preds):
            if (l == 1 and i != 0):
                aps.append(ap(query_labels, query_preds))
                query_preds = []
                query_labels = []

            query_preds.append(p)
            query_labels.append(l)

            i += 1

            if (i == len(preds)):
                aps.append(ap(query_labels, query_preds))
        return sum(aps) / float(len(aps))


    def compute_aps(preds, labels):
        aps = []
        query_preds = []
        query_labels = []
        i = 0
        for l, p in zip(labels, preds):
            if (l == 1 and i != 0):
                aps.append(ap(query_labels, query_preds))
                query_preds = []
                query_labels = []

            query_preds.append(p)
            query_labels.append(l)

            i += 1

            if (i == len(preds)):
                aps.append(ap(query_labels, query_preds))
        return aps

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name in ["quora", "mantis_10", "mantis_50", "ms_v2", "udc", "l4", "ms_marco_adhoc", "antique"]:
            return {"map": mean_average_precision(preds, labels), "ndcg_10": ndcg_at_10(preds, labels)}
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
