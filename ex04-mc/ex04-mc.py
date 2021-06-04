import gym
import numpy as np
import matplotlib.pyplot as plt
import random

env = gym.make('Blackjack-v0')

class MonteCarloEstimation(object):
    '''
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).

    '''

    def __init__(self):
        self.gamma = 1
        self.reset()

    def reset(self):
        # initialize value function 
        self.V = np.zeros((10, 10, 2)) # 12-21, 1-10
        self.N = np.zeros((10, 10, 2))

    def _convert_idx(self, s):
        #update
        p_idx = s[0]-12
        d_idx = s[1]-1
        a_idx = 1 if s[2] else 0
        return p_idx, d_idx, a_idx


    def policy_iteration(self, n_episodes, plot_n = 10000):
        #initilize
        self.reset()

        for i in range(n_episodes):
            state, action, rewards = self.run()
            # since in black jack we dont need to worry about seeing the states again - every visit == first visit
            G = 0

            for s, a ,r  in zip(state[::-1], action[::-1], rewards[::-1]):
                G = r + self.gamma*G

                #update
                p_idx, d_idx, a_idx = self._convert_idx(s)
                self.N[p_idx,d_idx,a_idx] += 1 

                N = self.N[p_idx,d_idx,a_idx]
                old_V = self.V[p_idx,d_idx,a_idx]
                self.V[p_idx,d_idx,a_idx] += (1/N)*(G - old_V)

            if i == plot_n:
                self.plot(plot_n)

        self.plot(n_episodes)

    def run(self):
        # runs on step of black jack

        history = [[] for _ in range(3)] # state ,action ,reward

        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        done = False
        while not done:
            history[0].append(obs)
            if obs[0] >= 20:
                obs, reward, done, _ = env.step(0)
                history[1].append(0)

            else:
                obs, reward, done, _ = env.step(1)
                history[1].append(1)

            history[2].append(reward)

        return history

    def plot(self, i):
        no_ace= self.V[:,:,0].flatten()
        ace= self.V[:,:,1].flatten()

        print(no_ace)
        X = []
        Y = []
        for x in range(12,22):
            for y in range(1,11):
                X.append(x)
                Y.append(y)

        assert(len(X) == len(no_ace))

        self.plot_helper(X, Y, no_ace, 'After {} Iterations - No Usable Ace'.format(i))
        self.plot_helper(X, Y, no_ace, 'After {} Iterations - Usable Ace'.format(i))

    def plot_helper(self, X, Y, A, name):

        fig = plt.figure() #figsize=plt.figaspect(0.5)
        ax = plt.axes(projection='3d')
        ax.set_zlim(-1.01, 1.01)
        surf = ax.plot_trisurf(Y, X, A, cmap=plt.cm.viridis, linewidth=0.1)
        fig.colorbar( surf, shrink=0.5, aspect=5)
        ax.set_title(name)
        ax.set_xlabel('Dear Showing')
        ax.set_ylabel('Player Hand')
        fig.savefig('fig{}.png'.format(name),dpi=300)
        plt.show()
        
class MonteCarloES(object):

    def __init__(self):
        self.gamma = 1
        self.reset()

    def reset(self):
        # initialize value function 
        self.Q = np.zeros((10, 10, 2, 2)) # 12-21, 1-10, usable ace, action (hit or stick)
        self.N = np.zeros((10, 10, 2, 2))
        self.P = np.zeros((10, 10, 2), dtype=np.uint8)

        #initialize with hitting when before 20
        self.P[:-2,:,:] =1 

    def _convert_idx(self, s):
        #update
        p_idx = s[0]-12
        d_idx = s[1]-1
        a_idx = 1 if s[2] else 0

        return p_idx, d_idx, a_idx

    def optimal_policy(self):
        self.P =  np.argmax(self.Q, axis=-1)

    def run(self):
        # runs one step of how the current greedy policy
        history = [[] for _ in range(3)] # state ,action ,reward

        # make random state-action to start with 
        obs = env.reset()
        action = random.randint(0,1)

        done = False
        while True:
            history[0].append(obs)
            history[1].append(action)
            obs, reward, done, _ = env.step(action)
            history[2].append(reward)

            #greedy action
            if done:
                break

            p_idx, d_idx, a_idx = self._convert_idx(obs)
            action = self.P[p_idx,d_idx,a_idx]

        return history


    def policy_es(self, n_episodes=1000000, verbose = 10000):
        #initilize
        self.reset()
        for i in range(n_episodes):
            state, action, rewards = self.run()
            # since in black jack we dont need to worry about seeing the states again - every visit == first visit
            G = 0

            for s, a ,r  in zip(state[::-1], action[::-1], rewards[::-1]):
                G = r + self.gamma*G

                #update
                p_idx, d_idx, a_idx = self._convert_idx(s)
                self.N[p_idx,d_idx,a_idx, a] += 1 

                N = self.N[p_idx,d_idx,a_idx, a]
                old_Q = self.Q[p_idx,d_idx,a_idx, a]
                self.Q[p_idx,d_idx,a_idx, a] += (1/N)*(G - old_Q)

                if i % verbose == 0:
                    self.print_detail(0,'At {} iterations: Policy when no usable ace'.format(i) )
                    self.print_detail(1,'At {} iterations: Policy when usable ace'.format(i) )

            #set policy greedy w.r.t Q
            self.optimal_policy()

    def print_detail(self,a, title):
        print(title)
        print('P\D',np.array([d for d in  range(1,11)]))
        for i in range(self.P.shape[0]):
            print(i+12, ':',self.P[i,:,a])
        print('\n')



def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    # MCE = MonteCarloEstimation()
    # MCE.policy_iteration(500000, 10000)

    MCES = MonteCarloES()
    MCES.policy_es(1000000)

    # print('Final Without usable ace')
    # print(MCES.Q[:,:,0,:])

    # print('With usable ace')
    # print(MCES.Q[:,:,1,:])

    # obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    # done = False
    # while not done:
    #     print("observation:", obs)
    #     if obs[0] >= 20:
    #         print("stick")
    #         obs, reward, done, _ = env.step(0)
    #     else:
    #         print("hit")
    #         obs, reward, done, _ = env.step(1)
    #     print("reward:", reward)
    #     print("")




if __name__ == "__main__":
    main()
