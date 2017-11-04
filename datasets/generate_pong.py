# A script to save Pong images to file
# Requires https://github.com/openai/gym#atari

import os
import json
from PIL import Image
import gym

ITERS = 20000

if not os.path.exists('/mnt/data/pong'):
    os.mkdir('/mnt/data/pong')
fp = open('/mnt/data/pong.dataset', 'w')

env = gym.make('Pong-v0')
obs = env.reset()
for i in range(10 * ITERS):
    img, reward, game_over, baz = env.step(env.action_space.sample())
    if game_over:
        env.reset()
    state = img[34:194:2,::2]  # The 160x160 square with the paddles, subsampled 2x2
    if i % 10 == 0:
        filename = '/mnt/data/pong/{:06d}.png'.format(i)
        Image.fromarray(state).save(filename)
        example = {'filename': filename, 'label': reward}
        fp.write(json.dumps(example) + '\n')
fp.close()

