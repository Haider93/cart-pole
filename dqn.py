import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
import scipy.stats
from q_agent import QAgent

EPISODES = 100


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=4))
        model.add(Activation('relu'))

        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future  = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(Q_future)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)


if __name__ == "__main__":
    # nor = scipy.stats.norm(0.0, 1.0)
    # print(nor.pdf(-4.5))

    env = gym.make('CartPole-v0')
    env._max_episode_steps = None

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    action_dist = []
    # multivariate
    # mvn = scipy.stats.multivariate_normal(np.zeros(state_size), np.eye(state_size, state_size))
    # action_dist.append(mvn)
    # action_dist.append(mvn)
    #univariate
    action_dist.append(scipy.stats.norm(0.0, 1.0))
    action_dist.append(scipy.stats.norm(0.0, 1.0))
    agent_q = QAgent(action_dist)

    def train():
        file = open('reward.csv','w')
        file.write("Episodes"+","+"reward"+"\n")

        # file = open('TD_Q_mvd.csv', 'w')
        # file.write("Episodes"+","+"time"+"\n")

        ##dqn agent

        # agent.load("model/cartpole-ddqn.h5")
        done = False
        batch_size = 128

        scores = []
        score = 0.0
        tot_reward = 0.0
        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            flag = 0
            #tot_reward = 0.0
            for time in range(1200):
                # uncomment this to see the actual rendering
                env.render()

                #dqn agent picks
                action = agent.act(state)

                ##qagent picks
                # state_normalized = agent_q.normalized_state(np.array(state))
                # action = agent_q.pick_action(agent_q.epsilon, action_dist, state_normalized)

                next_state, reward, done, info = env.step(action)
                # new_normalized_state = agent_q.normalized_state(np.array(next_state))
                #univariate
                # agent_q.TD_learning(new_normalized_state, state_normalized, reward, action)

                #multivariate
                #agent_q.TD_learning_mvd(new_normalized_state, state_normalized, reward, action)


                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                tot_reward += reward

                if done:
                    flag = 1
                    scores.append(time)
                    score += time
                    print("episode: {}/{}, score: {}, cum_reward :{}"
                          .format(e, EPISODES, score, tot_reward))
                    file.write(str(e)+","+str(tot_reward)+"\n")
                    #agent.update_target_model()
                    # print("episode: {}/{}, score: {}, e: {:.2}"
                    #       .format(e, EPISODES, time, agent.epsilon))
                    break
        file.close()
        agent_q.save()
        #agent.save("cartpole-dqn.h5")
        # file = open("test_TD1.csv", 'w+')
        # file.write("Episodes" + "," + "time_TD" + "\n")
        # params = agent_q.read()
        # action_dist_test = []
        # action_dist_test.append(scipy.stats.norm(float(params[0]), float(params[1])))
        # action_dist_test.append(scipy.stats.norm(float(params[2]), float(params[3])))
        # agent_q_test = QAgent(action_dist_test)
        # for e in range(20):
        #     state = env.reset()
        #     state = np.reshape(state, [1, state_size])
        #     flag = 0
        #     for time in range(1200):
        #         # uncomment this to see the actual rendering
        #         env.render()
        #
        #         #action = agent.act(state)
        #
        #         ##qagent picks
        #         state_normalized = agent_q_test.normalized_state(np.array(state))
        #         action = agent_q_test.pick_action(agent_q_test.epsilon, action_dist, state_normalized)
        #
        #         next_state, reward, done, info = env.step(action)
        #
        #         reward = reward if not done else -10
        #         next_state = np.reshape(next_state, [1, state_size])
        #         #agent.remember(state, action, reward, next_state, done)
        #         state = next_state
        #         if done:
        #             flag = 1
        #             scores.append(time)
        #             print("episode: {}/{}, score: {}"
        #                   .format(e, 20, time))
        #             file.write(str(e) + "," + str(time) + "\n")
        #             break
        #
        # file.close()



    ##test
    def test():
        file = open("test_TD_mvd.csv", 'w+')
        #agent = DQNAgent(state_size, action_size)
        #agent_q.load("cartpole-dqn.h5")

        params = agent_q.read()
        action_dist = []
        # action_dist.append(scipy.stats.norm(float(params[0]), float(params[1])))
        # action_dist.append(scipy.stats.norm(float(params[2]), float(params[3])))

        action_dist.append(scipy.stats.multivariate_normal(float(params[0]), float(params[1])))
        action_dist.append(scipy.stats.multivariate_normal(float(params[2]), float(params[3])))
        agent_q_test = QAgent(action_dist)
        scores = []
        for e in range(20):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            flag = 0
            for time in range(1200):
                # uncomment this to see the actual rendering
                env.render()

                #action = agent.act(state)

                ##qagent picks
                state_normalized = agent_q_test.normalized_state(np.array(state))
                action = agent_q_test.pick_action(agent_q_test.epsilon, action_dist, state_normalized)

                next_state, reward, done, info = env.step(action)

                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                #agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    flag = 1
                    scores.append(time)
                    print("episode: {}/{}, score: {}"
                          .format(e, 20, time))
                    file.write(str(e) + "," + str(time) + "\n")
                    break

        file.close()


            #
            # if len(agent.memory) > batch_size:
            #     agent.replay(batch_size)
        # if flag == 0:
        #     print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
        # if e % 100 == 0:
        #     print('saving the model')
        #     agent.save("model/cartpole-dqn.h5")
        #     # saving the figure
        #     plt.plot(scores)
        #     plt.savefig('score_plot')


    train()
    #test()