import numpy as np
import os

from environment import Pong

import matplotlib.pyplot as plt

os.makedirs("../graphic", exist_ok=True)

env = Pong()
env.reset()
for i in range(10):
    env.step(0)


observation = env.env.step(0)[0]
plt.subplot(1,4,1)
plt.imshow(observation)
plt.title('Original')

observation = observation[34:194,10:150,:]
observation = observation[::2,::2,:]
plt.subplot(1,4,2)
plt.imshow(observation)
plt.title('Resize')

observation = observation[:,:,0]
observation[observation == 144] = 0
observation[observation == 109] = 0
observation[observation != 0] = 1
plt.subplot(1,4,3)
plt.imshow(observation,cmap='gray')
plt.axis('off')
plt.title('Gray Scale')


o1 = env.step(0)[0]
o2 = env.step(2)[0]
state = (o2-o1 + 1)/2
plt.subplot(1,4,4)
plt.imshow(state,cmap='gray')
plt.axis('off')
plt.title('Residual State')

plt.gcf().set_size_inches(16,5)
plt.savefig("../graphic/preprocessing.jpg")
plt.clf()



reward = np.zeros((600,))
reward[-1] = 1
reward[149] = -1

plt.subplot(2,1,1)
plt.stem(reward)
plt.title('original')
plt.xlabel('epsode')
plt.ylabel('reward')

for i in range(599,-1,-1):
    if reward[i] != 0:
        r = reward[i]
    else:
        r *= 0.995
    reward[i] = r

plt.subplot(2,1,2)
plt.stem(reward)
plt.title('discounted')
plt.xlabel('epsode')
plt.ylabel('reward')
plt.gcf().set_size_inches(16,8)
plt.tight_layout()
plt.savefig("../graphic/reward.jpg")
plt.clf()
