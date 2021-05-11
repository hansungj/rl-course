import gym
import numpy as np
import sys

# Init environment
# env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
env = gym.make("FrozenLake-v0", desc=custom_map3x3)

# If you want to try larger maps you can do this using:
# random_map = gym.envs.frozen_lake.generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    terms = terminals()
    
    policy = np.zeros(n_states)
    steps = 0
    while True:
        steps +=1
        delta = 0
        for state in range(n_states):

            if state not in terms:
                max_backup = 0
                optimal_action = 0
                prev_value = V_states[state]

                for action in range(n_actions):
                    
                    backup = 0
                    for p, n_state, reward, terminal in env.P[state][action]:
                        if not terminal:
                            reward +=  gamma*V_states[n_state]
                        backup += p*reward

                    if backup > max_backup:
                        max_backup = backup
                        optimal_action = action

                #make greedy
                policy[state] = optimal_action
                V_states[state] = max_backup

                #update delta
                new_delta = abs(max_backup - prev_value)
                delta = max(new_delta, delta)

        if delta < theta:
            break

    print('Took {} steps'.format(steps))
    print('Final optimal value function is')
    print(V_states)
    return policy 

def terminals():
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms

def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
