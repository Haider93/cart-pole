import numpy as np
import random
import math
import scipy.stats
import matplotlib.pyplot as plt

class QAgent:
    def __init__(self, action_dist):
        self.action_dist = action_dist
        self.action_observation = np.zeros((2,), dtype=int)
        random.seed(10)
        self.alpha = 0.001
        self.gamma = 0.95
        self.epsilon = 0.8
        self.lamdaa = 0.85
        self.eligibility_trace = np.zeros((2,), dtype=float)
        self.vals = []

    def pick_action(self, epsilon, action_dist, state):
        if(random.uniform(0, 1) < epsilon):
            return random.randint(0, 1)
        action = 0
        max = action_dist[action].pdf(state)
        #for i in range(1, 1):
        action_value = action_dist[1].pdf(state)
        if max < action_value:
            action = 1
        return action

    def save(self):
        file = open("dist_params.txt", 'w+')
        # file.write(str(self.action_dist[0].mean())+","+str(self.action_dist[0].std())+"\n")
        # file.write(str(self.action_dist[1].mean()) + "," + str(self.action_dist[1].std()) + "\n")

        file.write(str(self.action_dist[0].mean) + "," + str(self.action_dist[0].cov) + "\n")
        file.write(str(self.action_dist[1].mean) + "," + str(self.action_dist[1].cov) + "\n")
        file.close()

    def read(self):
        with open("dist_params.txt") as fp:
            line = fp.readline()
            cnt = 1
            params =[]
            while line:
                print("Line {}: {}".format(cnt, line.strip()))
                line_content = line.split(',')
                line_content[1] = line_content[1].split('\n')
                params.append(line_content[0])
                params.append(line_content[1][0])
                line = fp.readline()
                cnt += 1
        return params
    def getQ(self, state, action):
        return self.action_dist[action].pdf(state)

    def getQMax(self, state):
        if self.action_dist[0].pdf(state) >= self.action_dist[1].pdf(state):
            return self.action_dist[0].pdf(state)
        else:
            return self.action_dist[1].pdf(state)

    def normalized_state(self, state):
        state = state.flatten()
        old_min = min(state)
        old_max = max(state)
        new_min = -1.0
        new_max = 1.0
        for i in range(0, len(state)):
            state[i] = new_min + (state[i]-old_min) * (new_max-new_min)/(old_max-old_min)
        return np.mean(state)

    def plotcur(self, val, action_dist):
        self.vals.append(val)
        if len(self.vals) == 5000:
            x = np.asarray(self.vals)
            plt.plot(x, scipy.stats.norm.pdf(x, action_dist[0].mean(), action_dist[0].std()), 'ro')
            plt.plot(x, scipy.stats.norm.pdf(x, action_dist[1].mean(), action_dist[1].std()), 'g^')
            #plt.plot(action_dist[0].mean(), action_dist[0].std())
            plt.show()

    def TD_learning(self, new_state, old_state, reward, action):
        prev_obs = self.action_observation[action]
        self.action_observation[action] = self.action_observation[action] + 1
        old_mean = self.action_dist[action].mean()
        if action == 0:
            self.eligibility_trace[action] = 1
            self.eligibility_trace[1] *= self.lamdaa * self.gamma
        else:
            self.eligibility_trace[action] = 1
            self.eligibility_trace[0] *= self.lamdaa * self.gamma

        #update = reward + self.gamma * self.getQ(new_state, self.pick_action(self.epsilon, self.action_dist, new_state)) - self.getQ(old_state, action)
        update = reward + self.gamma * self.getQMax(new_state) - self.getQ(old_state, action)
        update = self.alpha * update
        update *= (self.eligibility_trace[0] + self.eligibility_trace[1])
        new_mean = old_mean + ((update - old_mean)/self.action_observation[action])
        old_var = self.action_dist[action].var()
        new_var = (prev_obs / (prev_obs+1)) * (old_var + (pow((update - old_mean), 2.0) / (prev_obs + 1)))
        if new_var<0:
            exit(-1)
        if new_var == 0:
            new_var = old_var
        self.action_dist[action] = scipy.stats.norm(new_mean, math.sqrt(new_var))
        #self.plotcur(update, self.action_dist# )


    def TD_learning_mvd(self, new_state, old_state, reward, action):
        prev_obs = self.action_observation[action]
        self.action_observation[action] = self.action_observation[action] + 1
        old_mean = self.action_dist[action].mean
        if action == 0:
            self.eligibility_trace[action] = 1
            self.eligibility_trace[1] *= self.lamdaa * self.gamma
        else:
            self.eligibility_trace[action] = 1
            self.eligibility_trace[0] *= self.lamdaa * self.gamma

        #update = reward + self.gamma * self.getQ(new_state, self.pick_action(self.epsilon, self.action_dist, new_state)) - self.getQ(old_state, action)
        update = reward + self.gamma * self.getQMax(new_state) - self.getQ(old_state, action)
        update = self.alpha * update
        update *= (self.eligibility_trace[0] + self.eligibility_trace[1])
        new_mean = old_mean + ((update - old_mean)/self.action_observation[action])
        old_var = self.action_dist[action].cov
        new_var = (prev_obs / (prev_obs+1)) * (old_var + (pow((update - old_mean), 2.0) / (prev_obs + 1)))
        if np.linalg.det(new_var)<0:
            exit(-1)
        if np.linalg.det(new_var) == 0:
            new_var = old_var
        self.action_dist[action] = scipy.stats.multivariate_normal(new_mean, new_var)