# A script to save Pong images to file
# Requires https://github.com/openai/gym#atari

import os
import json
from PIL import Image
import gym

ITERS = 50000

if not os.path.exists('/mnt/data/space_invaders'):
    os.mkdir('/mnt/data/space_invaders')
fp = open('/mnt/data/space_invaders.dataset', 'w')

env = gym.make('SpaceInvaders-v0')
obs = env.reset()
for i in range(10 * ITERS):
    img, reward, game_over, foo = env.step(env.action_space.sample())
    if game_over:
        env.reset()
    state = img[40:200:2,::2]
    if i % 10 == 0:
        filename = '/mnt/data/space_invaders/{:06d}.png'.format(i)
        Image.fromarray(state).save(filename)
        fold = 'test' if i % 100 == 0 else 'train'
        if reward < 0:
            label = 'Bad'
        elif reward == 0:
            label = 'OK'
        else:
            label = 'Good'
        example = {'filename': filename, 'label': label, 'fold': fold}
        fp.write(json.dumps(example) + '\n')
fp.close()

