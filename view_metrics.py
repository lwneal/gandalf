# Print test_accuracy for each result directory
import os
import sys
import json
RESULTS_DIR = '/mnt/results'


def get_result_dirs():
    return [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]


def get_results(result_dir, epoch):
    filename = 'eval_epoch_{:04d}.json'.format(epoch)
    filename = os.path.join(RESULTS_DIR, result_dir, filename)
    if os.path.exists(filename):
        eval_folds = json.load(open(filename))
        return eval_folds
    return {}


def get_params(result_dir):
    filename = 'params.json'
    filename = os.path.join(RESULTS_DIR, result_dir, filename)
    return json.load(open(filename))


def get_dataset_name(result_dir):
    params = get_params(result_dir)
    dataset = params['dataset']
    return dataset.split('/')[-1].replace('.dataset', '')


def epoch_from_filename(filename):
    numbers = filename.split('epoch_')[-1].rstrip('.pth')
    return int(numbers)


def is_valid_directory(result_dir):
    result_dir = os.path.join(RESULTS_DIR, result_dir)
    if not os.path.exists(result_dir) or not os.path.isdir(result_dir):
        return False
    if 'params.json' not in os.listdir(result_dir):
        return False
    dirs = os.listdir(result_dir)
    if 'robotno' in dirs or 'norobot' in dirs:
        print("Found robotno in {}, skipping".format(result_dir))
        return False
    return True


def get_epochs(result_dir):
    filenames = os.listdir(os.path.join(RESULTS_DIR, result_dir))
    pth_names = [f for f in filenames if f.endswith('.pth')]
    return sorted(list(set(epoch_from_filename(f) for f in pth_names)))


def get_all_info(fold, metric):
    info = []
    for result_dir in get_result_dirs():
        # Evaluate the most recent epoch
        epochs = get_epochs(result_dir)
        if not epochs:
            continue
        epoch = epochs[-1]
        results = get_results(result_dir, epoch)
        params = get_params(result_dir)
        if not results:
            continue
        if fold not in results:
            continue
        info.append((result_dir, epoch, results[fold][metric], params))
    return info


if __name__ == '__main__':
    fold = 'test'
    metric = 'accuracy'
    if len(sys.argv) > 1:
        fold = sys.argv[1]
    if len(sys.argv) > 2:
        metric = sys.argv[2]

    infos = get_all_info(fold=fold, metric=metric)
    infos.sort(key=lambda x: x[0].split('_')[0] + str(x[2]))

    print('{:<64} {:>8} {:>8} {:>8}'.format("Experiment Name", "Epoch", fold + '_' + metric, "Superv."))
    for (name, epoch, metric, params) in infos:
        sys.stdout.write('{:<64} {:>8} {:>8.4f} {:>8}'.format(
            name, epoch, metric, str(params.get('supervised_encoder'))))
        if '--verbose' in sys.argv:
            for param in params:
                sys.stdout.write('\t{:>32}={:<32}'.format(param, params[param]))
        sys.stdout.write('\n')
