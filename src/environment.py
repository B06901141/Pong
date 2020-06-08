import gym

class Pong():
    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.reset()
    def step(self,action):
        observation, reward, done, _ = self.env.step(action)
        observation = self.preprocessing(observation)
        return observation, reward, done
    def reset(self):
        self.env.reset()
        for i in range(10):
            self.env.step(0)
    def render(self):
        return self.env.render()
    def preprocessing(self,observation):
        observation = observation[34:194,10:150,:]
        observation = observation[::2,::2,0]
        observation[observation == 144] = 0
        observation[observation == 109] = 0
        observation[observation != 0] = 1
        return observation #(80,70)

if __name__ == '__main__':
    env = Pong()
    print(env.step(1))
