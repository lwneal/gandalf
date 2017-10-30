import os
import json
import sys


def read_eval_file(filename):
    data = json.load(open(filename))
    return data


if __name__ == '__main__':
    eval_reports = []
    for arg in sys.argv[1:]:
        eval_reports.append(read_eval_file(arg))

    keys = eval_reports[0].keys()
    averages = {}
    for k in keys:
        values = [e[k]['accuracy'] for e in eval_reports]
        print(values)
        averages[k] = {'accuracy': sum(values) / len(values)}
    print(json.dumps(averages, indent=2, sort_keys=True))
