import time
import os
import numpy as np

def get_boltzman_distribution(q):
    (nS, nA) = q.shape
    boltzman_distribution = []
    for s in range(nS):
        boltzman_distribution.append([])
        for a in range(nA):
            boltzman_distribution[-1].append(np.exp(q[s][a]))
    boltzman_distribution = np.array(boltzman_distribution)
    boltzman_distribution /= np.sum(boltzman_distribution, axis=1).reshape(-1, 1)
    return np.array(boltzman_distribution)

###### Algorithm 1 ######

def iavi(feature_matrix, transition_probabilities, action_probabilities, trajectories, conf):

    # r = np.zeros((nS, nA))
    # q = np.zeros((nS, nA))
    # q_sh = np.zeros((nS, nA))

    boltzman_distribution = []
    evd_list = []
    q = None


    return q, evd_list, boltzman_distribution


###### Algorithm 2 ######

def iql(trajectories, conf):

    nS = conf['env']['grid_size']**2
    nA = conf['env']['num_actions']
    gamma = conf['env']['gamma']

    epochs = conf['iql']['epochs']
    alpha_r = conf['iql']['alpha_r']
    alpha_q = conf['iql']['alpha_q']
    alpha_sh = conf['iql']['alpha_sh']

    # initialize tables for reward function, value functions and state-action visitation counter.
    r = np.zeros((nS, nA))
    q = np.zeros((nS, nA))
    q_sh = np.zeros((nS, nA))
    state_action_visit = np.zeros((nS, nA))

    epsilon_for_log = 1e-10
    r_diff_list = []

    for i in range(epochs):
        if i%10 == 0:
            print("Epoch %s/%s" %(i+1, epochs))
       
        for traj in trajectories:
            for (s, a, _, ns) in traj:
                state_action_visit[s][a] += 1
                d = 0   # no terminal state

                # compute shifted q-function.
                q_sh[s, a] = (1-alpha_sh) * q_sh[s, a] + alpha_sh * (gamma * (1-d) * np.max(q[ns]))
                
                # compute log probabilities.
                state_action_visit_sum = np.sum(state_action_visit[s])
                log_prob = np.log((state_action_visit[s]/state_action_visit_sum) + epsilon_for_log)
                
                # compute eta_a and eta_b for Eq. (9).
                eta_a = log_prob[a] - q_sh[s][a]
                other_actions = [oa for oa in range(nA) if oa != a]
                eta_b = log_prob[other_actions] - q_sh[s][other_actions]
                sum_oa = (1/(nA-1)) * np.sum(r[s][other_actions] - eta_b)

                # update reward-function.
                r[s][a] = (1-alpha_r) * r[s][a] + alpha_r * (eta_a + sum_oa)

                # update value-function.
                q[s, a] = (1-alpha_q) * q[s, a] + alpha_q * (r[s, a] + gamma * (1-d) * np.max(q[ns]))
                s = ns
        

    boltzman_distribution = get_boltzman_distribution(q)
    
    return q, r_diff_list, boltzman_distribution