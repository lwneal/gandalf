# Delete all but the last checkpoint for each completed job
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
    try:
        return int(numbers)
    except ValueError:
        return 0


def is_valid_directory(result_dir):
    result_dir = os.path.join(RESULTS_DIR, result_dir)
    if not os.path.exists(result_dir) or not os.path.isdir(result_dir):
        return False
    if 'params.json' not in os.listdir(result_dir):
        return False
    dirs = os.listdir(result_dir)
    return True


def get_epochs(result_dir):
    filenames = os.listdir(os.path.join(RESULTS_DIR, result_dir))
    pth_names = [f for f in filenames if f.endswith('.pth')]
    return sorted(list(set(epoch_from_filename(f) for f in pth_names)))


def get_last_epoch(result_dir):
    return get_epochs(result_dir)[-1]


def get_all_pth(result_dir):
    results_dir = os.path.join(RESULTS_DIR, result_dir)
    filenames = [f for f in os.listdir(results_dir) if f.endswith('.pth')]
    return [os.path.join(RESULTS_DIR, result_dir, f) for f in filenames]


def check_result_dirs():
    to_delete = []
    for result_dir in get_result_dirs():
        epochs = get_epochs(result_dir)
        if not epochs:
            continue
        last_epoch = get_last_epoch(result_dir)
        print("Result dir {} has {} checkpoints saved, last checkpoint at epoch {}".format(
            result_dir, len(epochs), max(epochs)))
        for pth_file in get_all_pth(result_dir):
            if epoch_from_filename(pth_file) < last_epoch:
                to_delete.append(pth_file)
    return to_delete


def total_filesize(filename_list):
    bytes_count = 0
    for filename in filename_list:
        bytes_count += os.stat(filename).st_size
    return bytes_count


if __name__ == '__main__':
    to_delete = check_result_dirs()
    to_delete_bytes = total_filesize(to_delete)
    print("Found {} old files to delete totaling {} GB.".format(len(to_delete), to_delete_bytes // (2**30)))
    if '--force' in sys.argv:
        for filename in to_delete:
            os.remove(filename)
    else:
        print("Run with --force to delete the files")
