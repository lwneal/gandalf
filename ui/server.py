import flask
import json
import os
import random
from pprint import pprint


app = flask.Flask(__name__)

# TODO: Select a result_dir
RESULT_ID = 'mnist_28x28_8dim_unsupervised'

RESULT_DIR = '/mnt/results/{}'.format(RESULT_ID)
LABEL_DIR = os.path.join(RESULT_DIR, 'labels')
TRAJECTORY_DIR = os.path.join(RESULT_DIR, 'trajectories')


def save_active_label(label):
    if not os.path.exists(LABEL_DIR):
        print("Creating directory {}".format(LABEL_DIR))
        os.mkdir(LABEL_DIR)
    print("Saving label to {}".format(LABEL_DIR))
    pprint(label)
    filename = os.path.join(LABEL_DIR, '{}.json'.format(label['trajectory_id']))
    with open(filename, 'w') as fp:
        json.dump(label, fp, indent=2)
    return filename

def select_random_trajectory():
    # TODO: Instead of selecting a random trajectory, select an unlabeled one
    if not os.path.exists(TRAJECTORY_DIR):
        raise ValueError("Error: Trajectory directory {} does not exist")
    filenames = os.listdir(TRAJECTORY_DIR)
    filenames = [f for f in filenames if f.startswith('active') and f.endswith('.mp4')]
    if len(filenames) == 0:
        raise ValueError("Error: No active_learning .mp4 files available")
    filename = random.choice(filenames)
    return filename

@app.route('/')
def route_main_page():
    filename = select_random_trajectory()
    trajectory_id = filename.split('-')[-3]
    target_class = filename.split('-')[-1].replace('.mp4', '')
    start_class = filename.split('-')[-2].replace('.mp4', '')
    file_url = 'http://files.deeplearninggroup.com/{}/trajectories/{}'.format(RESULT_ID, filename)
    args = {
            'filename': filename,
            'file_url': file_url,
            'start_class': start_class,
            'target_class': target_class,
            'trajectory_id': trajectory_id,
    }
    return flask.render_template('index.html', **args)

@app.route('/static/<path:path>')
def serve_static():
    return send_from_directory('static', path)

@app.route('/submit', methods=['POST'])
def submit_value():
    print("Submitted value:")
    for k, v in flask.request.form.items():
        print("\t{}: {}".format(k, v))
    label = {
        'trajectory_id': flask.request.form['trajectory_id'],
        'start_class': flask.request.form['start_class'],
        'target_class': flask.request.form['target_class'],
        'label_point': flask.request.form['frame'],
    }
    save_active_label(label)
    return flask.redirect('/')

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
