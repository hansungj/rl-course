import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


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

def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e5)):
    Q = np.zeros((env.observation_space.n,  env.action_space.n))

    episode_lengths = []
    # TODO: implement the sarsa algorithm
    for i in tqdm(range(num_ep)):
        s=env.reset()
        a = epsilon_greedy(Q[s,:], epsilon)
        done=False
        j=0
        while not done:
            
            s_, r, done, _ = env.step(a)
            a_ = epsilon_greedy(Q[s_,:], epsilon)

            if not done:
                r += gamma*Q[s_, a_]

            #sarsa
            Q[s,a] = Q[s,a] + alpha*(r - Q[s,a])

            # update to next 
            a = a_
            s = s_

            j+=1 

            
        episode_lengths.append(j)

    return Q, episode_lengths

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


if __name__ == '__main__':
    #env=gym.make('FrozenLake-v0')
    env=gym.make('FrozenLake-v0', is_slippery=False)
    # env=gym.make('FrozenLake-v0', map_name="8x8")

    print("current environment: ")
    env.render()
    print("")

    print("Running sarsa...")
    # overall = np.zeros(int(1e4))
    # for i in tqdm(range(100)):
    #     Q, l = sarsa(env, num_ep = int(1e4))
    #     overall += np.array(l)

    # plt.plot(range(len(l)), overall/100)
    # plt.show()

    Q, l = sarsa(env, num_ep = int(1e4))    
    plot_V(Q, env)
    plot_Q(Q, env)
    print_policy(Q, env)
    plt.show()

    print("Running qlearning")
    Q = qlearning(env, num_ep = int(1e4))
    plot_V(Q, env)
    plot_Q(Q, env)
    print_policy(Q, env)
    plt.show()
