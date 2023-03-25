import numpy as np

def get_boltzman_distribution(q):
    (nS, nA) = q.shape
    bd = []
    for s in range(nS):
        bd.append([])
        for a in range(nA):bd[-1].append(np.exp(q[s][a]))
        
    bd = np.array(bd)
    action_sum = np.sum(bd, axis=1).reshape(-1, 1)
    bd = bd/action_sum
    return np.array(bd)

###### Algorithm 1 ######

def iavi(feature_matrix, transition_probabilities, action_probabilities, trajectories, conf):

    # r = np.zeros((nS, nA))
    # q = np.zeros((nS, nA))
    # q_sh = np.zeros((nS, nA))

    boltzman_distribution = []
    evd_list = []
    q = None

    # Need to be implemented


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

    r = np.zeros((nS, nA))
    q = np.zeros((nS, nA))
    q_sh = np.zeros((nS, nA))
    state_action_visit_counter = np.zeros((nS, nA))

    epsilon_for_log = 1e-4
    r_diff_list = []

    for i in range(epochs):
        if i%20 == 0:
            print(f"Epoch {i}")
       
        for traj in trajectories:
            for (s, a, _, new_s) in traj:

                state_action_visit_counter[s][a] += 1
        
                q_sh[s, a] = (1-alpha_sh) * q_sh[s, a] + alpha_sh * (gamma * np.max(q[new_s]))
                state_action_visit_sum = np.sum(state_action_visit_counter[s])
                log_prob = np.log((state_action_visit_counter[s]/state_action_visit_sum) + epsilon_for_log)
                other_a = [0,1,2,3,4].remove(a)

                eta_a = log_prob[a] - q_sh[s][a]
                eta_b = log_prob[other_a] - q_sh[s][other_a]
                sum_oa = (1/(nA-1)) * np.sum(r[s][other_a] - eta_b)
                r[s][a] = (1-alpha_r) * r[s][a] + alpha_r * (eta_a + sum_oa)
                q[s, a] = (1-alpha_q) * q[s, a] + alpha_q * (r[s, a] + gamma  * np.max(q[new_s]))
                s = new_s
        
    boltzman_distribution = get_boltzman_distribution(q)
    
    return q, r_diff_list, boltzman_distribution