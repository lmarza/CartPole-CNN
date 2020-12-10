import os
import random
import gym
import pylab
import numpy as np
from collections import deque

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



def Model(input_shape, action_space):
    input_x = Input(input_shape)
    input = input_x
    
    input = Conv2D(64, 5, strides=(3, 3),padding="valid", input_shape=input_shape, activation="relu", data_format="channels_first")(input)
    input = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="relu", data_format="channels_first")(input)
    input = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="relu", data_format="channels_first")(input)
    input = Flatten()(input)
    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    input = Dense(512, activation="relu", kernel_initializer='he_uniform')(input)

    # Hidden layer with 256 nodes
    input = Dense(256, activation="relu", kernel_initializer='he_uniform')(input)
    
    # Hidden layer with 64 nodes
    input = Dense(64, activation="relu", kernel_initializer='he_uniform')(input)

    # Output Layer with # of actions: 2 nodes (left, right)
    input = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(input)

    model = Model(inputs = input_x, outputs = input)
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    # export an image of model
    #dot_img_file = '/home/luca/Scrivania/model_1.png'
    #plot_model(model, to_file=dot_img_file, show_shapes=True)

    #model.summary()
    return model

class DQNAgent:
    def __init__(self, env_name):
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(0)  
       
        self.env._max_episode_steps = 1000
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 2000
        
        # Instantiate memory
        memory_size = 10000
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95    # discount rate
        
        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0 
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.0005  

        self.batch_size = 32
        self.TAU = 0.1 # target network soft update hyperparameter
        
        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): 
            os.makedirs(self.Save_Path)

        self.scores, self.episodes, self.average = [], [], []
        self.Model_name = os.path.join(self.Save_Path, self.env_name+"_DQN_CNN.h5")

        self.ROWS = 160
        self.COLS = 240
        self.FRAME_STEP = 4

        self.image_memory = np.zeros((self.FRAME_STEP, self.ROWS, self.COLS))
        self.state_size = (self.FRAME_STEP, self.ROWS, self.COLS)
        
        # create main model and target model
        self.model = Model(input_shape=self.state_size, action_space = self.action_size)
        self.target_model = Model(input_shape=self.state_size, action_space = self.action_size)  

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        q_model_theta = self.model.get_weights()
        target_model_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
            target_model_theta[counter] = target_weight
            counter += 1
        self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        self.memory.append((experience))

    def act(self, state, decay_step):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1-self.epsilon_decay)
        explore_probability = self.epsilon
    
        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            return np.argmax(self.model.predict(state)), explore_probability
                
    def replay(self):
        
        # Randomly sample minibatch from the deque memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size,) + self.state_size)
        next_state = np.zeros((self.batch_size,) + self.state_size)
        action, reward, done = [], [], []

        for i in range(len(minibatch)):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # predict Q-values for starting state using the main network
        target = self.model.predict(state)
        target_old = np.array(target)
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state)
        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)


    def imshow(self, image, frame_step=0):
        cv2.imshow("cartpole"+str(frame_step), image[frame_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def GetImage(self):
        img = self.env.render(mode='rgb_array')
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255

        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0,:,:] = img_rgb_resized

        #self.imshow(self.image_memory,0)        
        return np.expand_dims(self.image_memory, axis=0)

    def reset(self):
        self.env.reset()
        for i in range(self.FRAME_STEP):
            state = self.GetImage()
        return state

    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.GetImage()
        return next_state, reward, done, info
    
    def run(self):
        decay_step = 0
        for e in range(self.EPISODES):
            state = self.reset()
            done = False
            i = 0
            while not done:
                decay_step += 1
                action, explore_probability = self.act(state, decay_step)
                next_state, reward, done, _ = self.step(action)
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    if e % self.FRAME_STEP == 0:
                        self.update_target_model()
                    
                    self.scores.append(i)
                    self.episodes.append(e)
                    self.average.append(sum(self.scores[-100:]) / len(self.scores[-100:]))
            
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    if i == self.env._max_episode_steps:
                        print("Saving trained model to", self.Model_name)
                        #self.save(self.Model_name)
                        break
                self.replay()
        self.env.close()
        
    def test(self):
        self.load(self.Model_name)
        for e in range(self.EPISODES):
            state = self.reset()
            done = False
            i = 0
            while not done:
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = env.step(action)
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = DQNAgent(env_name)
    agent.run()
    #agent.test()
