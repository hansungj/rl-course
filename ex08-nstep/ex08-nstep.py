import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import random 
from tqdm import tqdm
'''
Implement n-step Sarsa and evaluate it on the 8x8 env. Evaluate the performance for different choices of n and alpha. Visualize your results 
plot the pe4foramcne over alpha for different choices of n simialr to lecture 8 slide 9

'''


def print_policy(Q, env):
    """ This is a helper function to print a nice policy from the Q function"""
    moves = [u'←', u'↓',u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in ['H', 'G']:
            policy[idx] = u'·'
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) 
        for row in policy]))


def plot_V(Q, env):
    """ This is a helper function to plot the state values from the Q function"""
    fig = plt.figure()
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
        if env.desc[idx] in ['H', 'G']:
            V[idx] = 0.
    plt.imshow(V, origin='upper', 
               extent=[0,dims[0],0,dims[1]], vmin=.0, vmax=.6, 
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y+0.5, dims[0]-x-0.5, '{:.3f}'.format(V[x,y]),
                horizontalalignment='center', 
                verticalalignment='center')
    plt.xticks([])
    plt.yticks([])


def plot_Q(Q, env):
    """ This is a helper function to plot the Q function """
    from matplotlib import colors, patches
    fig = plt.figure()
    ax = fig.gca()

    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape

    up = np.array([[0, 1], [0.5, 0.5], [1,1]])
    down = np.array([[0, 0], [0.5, 0.5], [1,0]])
    left = np.array([[0, 0], [0.5, 0.5], [0,1]])
    right = np.array([[1, 0], [0.5, 0.5], [1,1]])
    tri = [left, down, right, up]
    pos = [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5], [0.5, 0.8]]
    
    cmap = plt.cm.RdYlGn
    norm = colors.Normalize(vmin=.0,vmax=.6)
    
    ax.imshow(np.zeros(dims), origin='upper', extent=[0,dims[0],0,dims[1]], vmin=.0, vmax=.6, cmap=cmap)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        x, y = idx
        if env.desc[idx] in ['H', 'G']:
            ax.add_patch(patches.Rectangle((y, 3-x), 1, 1, color=cmap(.0)))
            plt.text(y+0.5, dims[0]-x-0.5, '{:.2f}'.format(.0),
                horizontalalignment='center', 
                verticalalignment='center')
            continue
        for a in range(len(tri)):
            ax.add_patch(patches.Polygon(tri[a] + np.array([y, 3-x]), color=cmap(Q[s][a])))
            plt.text(y+pos[a][0], dims[0]-1-x+pos[a][1], '{:.2f}'.format(Q[s][a]), 
                     horizontalalignment='center', verticalalignment='center',
                    fontsize=9, fontweight=('bold' if Q[s][a] == np.max(Q[s]) else 'normal'))

    plt.xticks([])
    plt.yticks([])

def epsilon_greedy(A, eps):
    u = random.random()
    if u > eps:
        return random.choice(np.flatnonzero(A == A.max()))
    return np.random.randint(0,len(A)-1)

def qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e5)):
    Q = np.zeros((env.observation_space.n,  env.action_space.n))
    # TODO: implement the qlearning algorithm

    for i in tqdm(range(num_ep)):
        s=env.reset()
        done=False
        while not done:
            a = epsilon_greedy(Q[s,:], epsilon)
            s_, r, done, _ = env.step(a)

            if not done:
                r += gamma*np.argmax(Q[s_,:])

            #q learing 
            Q[s,a] = Q[s,a] + alpha*(r - Q[s,a])

            # update to next 
            s = s_
    return Q

def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    
    #init Q 
    Q = np.zeros((env.observation_space.n,  env.action_space.n))
    
    for _ in tqdm(range(num_ep)):
        s = env.reset()
        a = epsilon_greedy(Q[s,:], epsilon)
        done=False
        t=0
        T=float('inf')
        memory = [(s,a)] # we keep track of only last n state-action paris 
        rewards = [] # we keep track of only last n rewards, starting from +1
        while True:
            if t < T: 
                a = memory[0][1]
                s_, r, done, _ = env.step(a)
                a_ = epsilon_greedy(Q[s_,:], epsilon)
                
                rewards.append(r)
                memory.append((s_,a_))
                if done:
                    T = t+1

            tau = t - n + 1 
            if tau >= 0:
                # calculate the returns 
                G = sum((gamma**i)*reward for i, reward in enumerate(rewards))
                if tau + n < T:
                    G += (gamma**n)*Q[memory[-1]]
                Q[memory[0]] += alpha*(G  - Q[memory[0]])

            if tau > T-1:
                break

            if len(memory) == n+1 : # keep only the last n in memory 
                memory = memory[1:]
                rewards = rewards[1:] # but we have n rewards not n+1
                
            t += 1
    return Q


def rms(Q,V_true):
    #we use the value which we acquire from the brutefforce method 
    #reduce our Q to to V
    V = np.max(Q, dim=-1)
    N = len(V)
    rms = np.sum(np.sqrt((V - V_true)**2))/N
    return rms 

env=gym.make('FrozenLake-v0', map_name="8x8")

alphas = [0.1*i for i in range(9)]
N = [i for i in range(20)]
num_ep=int(1e4)

Qsummary = np.zeros((len(alphas), len(N)))
for i, n in enumerate(N):
    for j, alpha in enumerate(alphas):
        Q = nstep_sarsa(env, n=n,alpha=alpha )
        Qsummary[i,j] = rms(Q)
    plt.plot(Q[i,:], label='n={}'.format(i))

plt.ylabel('average RMS averaged over all state first {} episodes'.format(num_ep))
plt.xlabe('alpha')
plt.show()