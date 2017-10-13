import flask
import json
import os
import random
from pprint import pprint


FILES_URL = 'http://files.deeplearninggroup.com'
RESULTS_PATH = '/mnt/results/'
app = flask.Flask(__name__)


def save_active_label(label, result_dir):
    result_dir = os.path.join(RESULTS_PATH, result_dir)
    label_dir = os.path.join(result_dir, 'labels')
    if not os.path.exists(label_dir):
        print("Creating directory {}".format(label_dir))
        os.mkdir(label_dir)
    print("Saving label to {}".format(label_dir))
    pprint(label)
    filename = os.path.join(label_dir, '{}.json'.format(label['trajectory_id']))
    with open(filename, 'w') as fp:
        json.dump(label, fp, indent=2)
    return filename


def is_labeled(filename, result_dir):
    key = filename.split('-')[1]
    labels = os.path.join(result_dir, 'labels')
    if not os.path.exists(labels):
        print("Labels directory does not exist, creating it")
        os.mkdir(labels)
    label_keys = [l.replace('.json', '') for l in os.listdir(labels)]
    return key in label_keys


def get_counts(result_dir):
    result_dir = os.path.join(RESULTS_PATH, result_dir)
    trajectories_dir = os.path.join(result_dir, 'trajectories')
    labels_dir = os.path.join(result_dir, 'labels')
    
    if os.path.isdir(trajectories_dir):
        trajectory_count = len([f for f in os.listdir(trajectories_dir) if f.endswith('.npy')])
    else:
        trajectory_count = 0

    if os.path.isdir(labels_dir):
        label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.json')])
    else:
        label_count = 0
    return trajectory_count, label_count


def get_unlabeled_trajectories(result_dir, fold='active'):
    result_dir = os.path.join(RESULTS_PATH, result_dir)
    if not os.path.exists(result_dir):
        raise ValueError("Could not load result directory {}".format(result_dir))

    trajectory_dir = os.path.join(result_dir, 'trajectories')
    if not os.path.exists(trajectory_dir):
        print("Trajectory directory {} does not exist, creating it".format(trajectory_dir))
        os.mkdir(trajectory_dir)

    filenames = os.listdir(trajectory_dir)
    filenames = [f for f in filenames if f.startswith(fold) and f.endswith('.mp4')]
    if len(filenames) == 0:
        print("No active_learning .mp4 files available")
        return []

    unlabeled_filenames = [f for f in filenames if not is_labeled(f, result_dir)]
    if len(unlabeled_filenames) == 0:
        print("All {} trajectories have been labeled".format(len(unlabeled_filenames)))
    return unlabeled_filenames


def get_result_dirs():
    return [f for f in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, f))]


def get_args_active(filename, result_dir):
    trajectory_id = filename.split('-')[-3]
    target_class = filename.split('-')[-1].replace('.mp4', '')
    start_class = filename.split('-')[-2].replace('.mp4', '')
    file_url = '{}/{}/trajectories/{}'.format(FILES_URL, result_dir, filename)
    return {
            'result_dir': result_dir,
            'filename': filename,
            'file_url': file_url,
            'start_class': start_class,
            'target_class': target_class,
            'trajectory_id': trajectory_id,
    }


def get_args_batch(filename, result_dir):
    trajectory_id = filename.split('-')[-3]
    target_class = filename.split('-')[-1].replace('.mp4', '')
    start_class = filename.split('-')[-2].replace('.mp4', '')
    file_url = '{}/{}/trajectories/{}'.format(FILES_URL, result_dir, filename)
    return {
            'result_dir': result_dir,
            'filename': filename,
            'file_url': file_url,
            'start_class': start_class,
            'target_class': target_class,
            'trajectory_id': trajectory_id,
    }


@app.route('/')
def route_main_page():
    table_contents = []
    for result_dir in get_result_dirs():
        row = {
            'result_dir': result_dir
        }
        row['trajectory_count'], row['labeled_count'] = get_counts(result_dir)
        row['unlabeled_count'] = row['trajectory_count'] - row['labeled_count']
        table_contents.append(row)
    args = {
            'table_contents': table_contents,
            'result_count': len(table_contents),
    }
    return flask.render_template('index.html', **args)


@app.route('/active/<result_dir>')
def route_label_slider(result_dir):
    filenames = get_unlabeled_trajectories(result_dir, fold='active')
    if filenames:
        filename = random.choice(filenames)
        args = get_args_active(filename, result_dir)
        args['unlabeled_count'] = len(filenames)
    else:
        args = {'unlabeled_count': 0}
    return flask.render_template('label_slider.html', **args)


@app.route('/batch/<result_dir>')
def route_label_batch(result_dir):
    filenames = get_unlabeled_trajectories(result_dir, fold='batch')
    if filenames:
        filename = random.choice(filenames)
        args = get_args_batch(filename, result_dir)
        args['unlabeled_count'] = len(filenames)
    else:
        args = {'unlabeled_count': 0}
    return flask.render_template('label_batch.html', **args)


@app.route('/submit/<result_dir>', methods=['POST'])
def submit_value(result_dir):
    label = {
        'trajectory_id': flask.request.form['trajectory_id'],
        'start_class': flask.request.form['start_class'],
        'target_class': flask.request.form['target_class'],
        'label_point': flask.request.form['frame'],
    }
    save_active_label(label, result_dir)
    return flask.redirect(flask.request.referrer)


@app.route('/static/<path:path>')
def serve_static():
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
