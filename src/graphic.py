import numpy as np
import os

import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.makedirs("../graphic", exist_ok=True)
    smooth = lambda a,k: np.convolve(a,np.ones((15,))/15, 'valid') if k == 1 else smooth(np.convolve(a,np.ones((15,))/15, 'valid'), k-1)
    with open("../log/history.csv", "r") as f:
        history = f.read()
    history = history.split("\n")[1:-1]
    history = [i.split(',') for i in history]
    history = np.array(history,dtype = np.int32)
    smoothed = smooth(history[:,2],1)
    plt.plot(history[:,2])
    plt.plot(smoothed)
    plt.axhline(0,c = 'k')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training History")
    plt.gcf().set_size_inches(16,8)
    plt.savefig("../graphic/history.jpg")
