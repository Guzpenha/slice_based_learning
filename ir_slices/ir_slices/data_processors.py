import os
import csv
import sys
import copy
import json

from typing import List
from tqdm import tqdm
from IPython import embed

csv.field_size_limit(sys.maxsize)

def transform_to_q_docs_format(examples):
    """
    Groups examples of each query.
    BE CAREFUL! This function assumes that every query has only one relevant document.
    """
    doc_q_examples = []
    docs = []
    labels = []
    for i, example in enumerate(examples):
        if example[2] == "1" and i!=0:
            doc_q_examples.append([examples[i-1][0], docs, labels])
            docs = []
            labels = []
        docs.append(example[1])
        labels.append(example[2])

        if i == (len(examples)-1):
            doc_q_examples.append([example[0], docs, labels])
    return doc_q_examples

class InputExample(object):
    """
    A single training/test example for a retrieval task

    Inputs
    =========
    q_id: unique identifier for the instance
    query: a list containing the query texts, in conversational response ranking len(query)>=1
    documents: a list containing the candidate documents to rank
    label: a list with the relevance of the query wrt each document

    """
    def __init__(self, instance_id: str, query: List[str],
                 documents: List[str], labels: List[str]=None):
        self.instance_id = instance_id
        self.query = query
        self.documents = documents
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcessor(object):
    """Base class for data converters for retrieval tasks"""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class CRRProcessor(DataProcessor):
    """Processor for the Conversation Response Ranking datasets, such as MSDialog, UDC and MANtIS."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        q_doc_lines = []
        for (i, line) in enumerate(lines):
            query = line[1:-1]
            doc = line[-1][0:-2]
            label = line[0]
            q_doc_lines.append([query, doc, label])
        lines_by_q = transform_to_q_docs_format(q_doc_lines)
        examples = []
        for i, line in enumerate(lines_by_q):
            id = "%s-%s" % (set_type, i)
            examples.append(InputExample(instance_id=id, query=line[0], documents=line[1], labels=line[2]))
        return examples

class QuoraProcessor(DataProcessor):
    """Processor for the QUORA data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "quora_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "quora_dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "quora_test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        q_doc_lines = []
        for (i, line) in enumerate(lines):
            query = line[3]
            doc = line[4]
            if line[0] == "0":
                label = "1"
            else:
                label = "0"
            q_doc_lines.append([query, doc, label])
        lines_by_q = transform_to_q_docs_format(q_doc_lines)
        examples = []
        for i, line in enumerate(lines_by_q):
            id = "%s-%s" % (set_type, i)
            examples.append(InputExample(instance_id=id, query=[line[0]], documents=line[1], labels=line[2]))
        return examples

class QAProcessor(DataProcessor):
    """Processor for the L4 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        q_doc_lines = []
        for (i, line) in enumerate(lines):
            label = line[0]
            query = line[1]
            doc = line[2]
            q_doc_lines.append([query, doc, label])
        lines_by_q = transform_to_q_docs_format(q_doc_lines)
        examples = []
        for i, line in enumerate(lines_by_q):
            id = "%s-%s" % (set_type, i)
            examples.append(InputExample(instance_id=id, query=[line[0]], documents=line[1], labels=line[2]))
        return examples

processors = {
    "quora": QuoraProcessor,
    "mantis_10": CRRProcessor,
    "mantis_50": CRRProcessor,
    "ms_v2": CRRProcessor,
    "udc": CRRProcessor,
    "l4": QAProcessor,
    "antique": QAProcessor,
    "ms_marco_adhoc": QAProcessor
}