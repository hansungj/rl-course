import gym
import numpy as np
import itertools 
from tqdm import tqdm 
# Init environment
# Lets use a smaller 3x3 custom map for faster computations
# custom_map3x3 = [
#     'SFF',
#     'FFF',
#     'FHG',
# ]
# env = gym.make("FrozenLake-v0", desc=custom_map3x3)
env=gym.make('FrozenLake-v0', map_name="8x8")
# TODO: Uncomment the following line to try the default map (4x4):
# custom_map4x4 = [
#     'SFHF',
#     'FFFF',
#     'FFFG',
# ]
# env = gym.make("FrozenLake-v01",desc=custom_map4x4)

# Uncomment the following lines for even larger maps:
#random_map = generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

r = np.zeros(n_states) # the r vector is zero everywhere except for the goal state (last state)
r[-1] = 1.

gamma = 0.8


""" This is a helper function that returns the transition probability matrix P for a policy """
def trans_matrix_for_policy(policy):
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions


""" This is a helper function that returns terminal states """
def terminals():
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy):
    P = trans_matrix_for_policy(policy)
    # TODO: calculate and return v
    # (P, r and gamma already given)
    # (I -  gamma*P)v = r  


    P = trans_matrix_for_policy(policy)
    v = np.linalg.inv(np.identity(n_states) - gamma*P)@r
    return v


def bruteforce_policies():
    terms = terminals()
    optimalpolicies = []

    policy = np.zeros(n_states, dtype=np.int)  # in the discrete case a policy is just an array with action = policy[state]
    optimalvalue = np.zeros(n_states)
    
    # TODO: implement code that tries all possible policies, calculate the values using def value_policy. Find the optimal values and the optimal policies to answer the exercise questions.
    all_possible_policies = itertools.product(range(n_actions), repeat=n_states- len(terms)) 
 
    for policy_cd in tqdm(all_possible_policies):

        #add in terminals
        j = 0
        for i in range(len(policy)):
            if i in terms:
                continue
            policy[i] = policy_cd[j] 
            j+=1

        value = value_policy(policy)

        if (value >= optimalvalue).all():
            if (value == optimalvalue).all():
                optimalpolicies.append(list(policy))
                continue

            optimalvalue = value
            optimalpolicies = [list(policy)]



    print ("Optimal value function:")
    print(optimalvalue)
    print ("number optimal policies:")
    print (len(optimalpolicies))
    print ("optimal policies:")
    print (np.array(optimalpolicies))
    return optimalpolicies



def main():
    # print the environment

    print("current environment: ")
    env.render()
    print("")

    # # Here a policy is just an array with the action for a state as element
    # policy_left = np.zeros(n_states, dtype=np.int)  # 0 for all states
    # policy_right = np.ones(n_states, dtype=np.int) * 2  # 2 for all states

    # # Value functions:
    # print("Value function for policy_left (always going left):")
    # print (value_policy(policy_left))
    # print("Value function for policy_right (always going right):")
    # print (value_policy(policy_right))

    optimalpolicies = bruteforce_policies()
    np.save('bruteforce_optimal.npy', value_policy(optimalpolicies[0]))

    # This code can be used to "rollout" a policy in the environment:
    """
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(optimalpolicies[0][state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
