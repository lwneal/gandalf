# A script to save Pong images to file
# Requires https://github.com/openai/gym#atari

from tqdm import tqdm
import numpy as np
import os
import json
from PIL import Image
import gym

ITERS = 500000
BACKGROUND_COLOR = [144,72,17]
PADDLE_COLOR = 213
LEFT_PADDLE_COL = 9
DATASET_NAME = 'pong-random'
DIR_NAME = os.path.join('/mnt/data', DATASET_NAME)
TRAIN_TEST_SPLIT = 10
np.random.seed(42)


def get_left_paddle_height(state):
    left_paddle_col = state[:,9,0]
    left_paddle_row = np.argwhere(left_paddle_col == PADDLE_COLOR)
    if len(left_paddle_row) == 0:
        return None
    return left_paddle_row[0,0]


def get_label_for_state(state):
    height = get_left_paddle_height(state)
    if height is None:
        return None
    return 'Bottom' if height >= 36 else 'Top'


def ball_exists(state):
    return state[:,:,0].max() == 236


def step(env):
    img, reward, game_over, foo = env.step(env.action_space.sample())
    if game_over:
        env.reset()
    # The 160x160 square with the paddles, subsampled 2x2
    play_field_state = img[34:194:2,::2]  
    return play_field_state


def main():
    if not os.path.exists(DIR_NAME):
        os.mkdir(DIR_NAME)

    fp = open('/mnt/data/{}.dataset'.format(DATASET_NAME), 'w')
    env = gym.make('Pong-v0')
    obs = env.reset()
    for i in tqdm(range(ITERS)):
        state = step(env)
        if not ball_exists(state):
            continue
        label = get_label_for_state(state)
        if label is None:
            continue
        filename = os.path.join(DIR_NAME, '{:06d}.png'.format(i))
        Image.fromarray(state).save(filename)
        fold = 'test' if i % TRAIN_TEST_SPLIT == 0 else 'train'
        example = {
            'filename': filename,
            'label': label,
            'fold': fold,
        }
        fp.write(json.dumps(example) + '\n')
    fp.close()

if __name__ == '__main__':
    main()
