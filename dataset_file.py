"""
 A Dataset file has the extension .dataset
 If it begins with the two byte sequence 0x1F8B then it is compressed
 Otherwise it is plain-text
 To decompress it, cat filename.dataset | gzip
 A decompressed .dataset file is a newline-delimited text file 
 Each line is a JSON key-value dictionary object.
  {"filename": "foo/bar.jpg", "baz": 1}
  {"filename": "foo/boo.jpg", "baz": -1}
  {"filename": "baz/foo.jpg", "baz": 2, "color": "blue"}
 By convention, "filename" is a relative path to a JPG/PNG file in the DATA_DIR directory
 Any boolean property should start with "is_" or "has_"
"""
import os
import json
import numpy as np
import imutil
import copy


class DatasetFile(object):
    def __init__(self, input_filename):
        input_filename = os.path.expanduser(input_filename)
        self.data_dir = os.path.dirname(input_filename)
        self.name = os.path.split(input_filename)[-1].replace('.dataset', '')

        data = open(input_filename).read()
        if data.startswith(chr(0x1F) + chr(0x8B)):
            print("Decompressing gzip file size {}".format(len(data)))
            data = data.decode('zlib')
        lines = data.splitlines()
        self.examples = [json.loads(l) for l in lines]
        print("Dataset {} contains {} examples:".format(self.name, len(self.examples)))
        self.folds = get_folds(self.examples)
        for name, count in self.folds.items():
            print("\tFold '{}': {} examples".format(name, count))

    def __add__(self, other):
        summed = copy.copy(self)
        for other_example in other.examples:
            summed.examples.append(other_example)
        summed.name = '-'.join([self.name, other.name])
        print("Combined dataset {} contains {} examples".format(
            summed.name, len(summed.examples)))
        return summed

    def _random_idx(self):
        return np.random.randint(0, len(self.examples))

    def count(self):
        return len(self.examples)

    def get_example(self, requirements=None, fold=None):
        # TODO: Index instead of rejection sampling
        while True:
            example = self.examples[self._random_idx()]
            if fold and fold != example.get('fold'):
                continue
            if requirements and not all(r in example for r in requirements):
                continue
            return example

    def get_all_examples(self, requirements=None, fold=None):
        for example in self.examples:
            if fold and fold != example.get('fold'):
                continue
            if requirements and not all(r in example for r in requirements):
                continue
            yield example

    def get_batch(self, requirements=None, fold=None, batch_size=16):
        examples = []
        for i in range(batch_size):
            examples.append(self.get_example(requirements, fold))
        return examples


def get_folds(examples):
    items_per_fold = {}
    for e in examples:
        fold = e.get('fold')
        if not fold:
            continue
        if fold not in items_per_fold:
            items_per_fold[fold] = 0
        items_per_fold[fold] += 1
    return items_per_fold
