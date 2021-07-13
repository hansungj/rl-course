import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

'''
Question 1. 
Count state-action pair occurancies in the expert demonstrations and construct a policy from that. (2P)

Question 2. 
Use a linear reward representation and one-hot encoded features. A function for computing the optimal policy is
already given in the template. Compute state frequencies p(s|ψ) as given on slide 20.

'''
def generate_demonstrations(env, expertpolicy, epsilon=0.1, n_trajs=100):
    """ This is a helper function that generates trajectories using an expert policy """
    demonstrations = []
    for d in range(n_trajs):
        traj = []
        state = env.reset()
        for i in range(100):
            if np.random.uniform() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = expertpolicy[state]
            traj.append((state, action))  # one trajectory is a list with (state, action) pairs
            state, _, done, info = env.step(action)
            if done:
                traj.append((state, 0))
                break
        demonstrations.append(traj)
    return demonstrations  # return list of trajectories


def plot_rewards(rewards, env):
    """ This is a helper function to plot the reward function"""
    fig = plt.figure()
    dims = env.desc.shape
    plt.imshow(np.reshape(rewards, dims), origin='upper', 
               extent=[0,dims[0],0,dims[1]], 
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y+0.5, dims[0]-x-0.5, '{:.3f}'.format(np.reshape(rewards, dims)[x,y]),
                horizontalalignment='center', 
                verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def count_sa_occupancies(S, A, trajs):
    policy = np.zeros((S,A))
    for traj in trajs:
        for state, action in traj:
            policy[state, action] += 1
    
    # normalize 
    policy =  policy / np.sum(policy, axis=1, keepdims=True)

    # if we have division by zero we just replace with random 
    nans = np.isnan(policy)
    policy[nans] = 1/A

    return np.argmax(policy, axis=1)

class MaximumEntropyRL:

    def __init__(self, env, alpha = 0.01):
        
        # linear reward
        self.env = env  
        self.n_S = self.env.observation_space.n
        self.n_A = self.env.action_space.n
        self.alpha = alpha 
    
    def compute_state_frequency(self, policy, T=100):
        '''
        mu_{t+1}(s) <- \sum_s \sum_a p(s'|s,a)pi(a|s)mu_{t}(s)
        '''

        mus = np.zeros((T, self.n_S))
        mus[0,0] = 1 # start state
        transitions = self.trans_matrix_for_policy(policy)

        for i in range(T-1):
            mus[i+1, :] = transitions.T@mus[i,:]
        mu= np.sum(mus, axis=0) / T
        return mu
    
    def trans_matrix_for_policy(self, policy):
        transitions = np.zeros((self.n_S, self.n_S))
        for s in range(self.n_S):
            probs = self.env.P[s][policy[s]]
            for el in probs:
                transitions[s, el[1]] += el[0]
        return transitions

    def value_iteration(self):
        """ Computes a policy using value iteration given a list of rewards (one reward per state) """
        rewards = self.w # rewards is just our weight w because we use a linear representation 
        V_states = np.zeros(self.n_S)
        theta = 1e-8
        gamma = .9
        maxiter = 1000
        policy = np.zeros(self.n_S, dtype=np.int32)
        for iter in range(maxiter):
            delta = 0.
            for s in range(self.n_S):
                v = V_states[s]
                v_actions = np.zeros(self.n_A) # values for possible next actions
                for a in range(self.n_A):  # compute values for possible next actions
                    v_actions[a] = rewards[s]
                    for tuple in self.env.P[s][a]:  # this implements the sum over s'
                        v_actions[a] += tuple[0]*gamma*V_states[tuple[1]]  # discounted value of next state
                policy[s] = np.argmax(v_actions)
                V_states[s] = np.max(v_actions)  # use the max
                delta = max(delta, abs(v-V_states[s]))

            if delta < theta:
                break
        return policy
    
    def train(self, trajs, num_iter=100):
        '''
        maximum entropy inverse reinforcement learning algorithm 

        the objective of MEIRL is to maximize the probabilities of trajectories from the demo 
        J = \sum_{i=1}^N log p(R(tau_i))

        in order to comptue the gradient of this log likelihood 
        \nabla_J = \sum_{i=1}^N \nabla R(tau) - |D|E_{tau ~ sub_opt_pi}[ \nabla R(tau) ]

        and 
        \nabla_J = \sum_{i=1}^N \nabla R(tau) - |D| \sum_{j=1}^M \nabla R(tau_j)

        but if we know the dynamics 
        \nabla_J = \sum_{i=1}^N \nabla R(tau) - |D| \sum_s pi(s|psi) \nabla r(s)

        since the reward function is a linear function 
        \nabla_J = \sum_{i=1}^N  \sum_t \nabla r(s_t) - |D| mu(s|psi)
        
        '''
        #step 1 initialize reward and demo 
        self.init_reward_function(num_iter)
        N = len(trajs)

        for i in range(num_iter):
            #step 2 compute optimal policy 
            policy = self.value_iteration()

            #step 3 compute state occupancy frequency 
            mu = self.compute_state_frequency(policy)

            #step 4 compute gradient 
            #step 4.1 compute the empirical (expert) expectation of the trajctories 
            emp = np.zeros(self.n_S)
            for traj in trajs:
                for state, _ in traj:
                    emp[state] += self.w[state]
            
            gradient =(1/N)*emp
            
            # step 4.2 compute model expectation of the trajectoreis - which is just the state occupancy  
            self.gradient -= mu

            # step 5 gradient step
            self.gradient_step()

        #compute final policy
        policy = self.value_iteration()
        return policy

    def gradient_step(self):
        self.w += self.alpha*self.gradient
    
    def init_reward_function(self, n_iter=None):
        self.w = np.zeros(self.n_S)
        self.gradient = np.zeros(self.n_S)

def main():
    env = gym.make('FrozenLake-v0')
    #env.render()
    env.seed(0)
    np.random.seed(0)
    expertpolicy = [0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    trajs = generate_demonstrations(env, expertpolicy, 0.1, 20)  # list of trajectories
    print("one trajectory is a list with (state, action) pairs:")
    print (trajs[0])

    # Question 1 
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = count_sa_occupancies(n_states, n_actions, trajs)
    print('\nQuestion 1 : the extracted policy is:')
    print(policy)

    #Question 2 - 3 
    meRL = MaximumEntropyRL(env)
    policy =meRL.train(trajs,num_iter=100)

    print('\nQuestion 2-3 : implement inverse RL - the acquired policy is ')
    print(policy)
    print('\nThe actual expert policy is')
    print(expertpolicy)
    print('\nAcquired reward funcxtion is :')
    print(meRL.w)
    # plot_rewards(meRL.w, env)

if __name__ == "__main__":
    main()
