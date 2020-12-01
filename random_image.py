import gym
import random
import numpy as np
import cv2

class DQN_CNN_Agent:
    def __init__(self, env_name):
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.ROWS = 160
        self.COLS = 240
        self.REM_STEP = 4
        self.EPISODES = 10
        #remember 4 frame of size 160X240
        self.image_memory = np.zeros((self.REM_STEP, self.ROWS, self.COLS))

    def imshow(self, image, rem_step=0):
        cv2.imshow(env_name+str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def getImage(self):
        img = self.env.render(mode='rgb_array')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #resize images for CNN input
        img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255

        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0,:,:] = img_rgb_resized

        self.imshow(self.image_memory,0)
        
        return np.expand_dims(self.image_memory, axis=0)
    
    def reset(self):
        self.env.reset()
        for i in range(self.REM_STEP):
            state = self.getImage()
        return state

    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.GetImage()
        return next_state, reward, done, info

    def run(self):
        for episode in range(self.EPISODES):
            self.reset()
            for t in range(500):               
                action = self.env.action_space.sample()
                next_state, reward, done, info = self.step(action)
                if done:
                    break

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = DQN_CNN_Agent(env_name)
    agent.run()
