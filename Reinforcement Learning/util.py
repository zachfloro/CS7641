import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go

def get_transitions(env):
    P = np.zeros(shape=(env.action_space.n, env.observation_space.n, env.observation_space.n))
    
    for a in range(env.action_space.n):
        for s in range(env.observation_space.n):
            for p, s_prime, r, done in env.P[s][a]:
                P[a,s,s_prime] += p
    
    return P

def get_transitions_forest(P, perc):
    inc = perc/len(P[0])
    perc = perc/len(P[0])
    for i in range(len(P[0])):
        if i < 9:
            P[0][i][0] -= perc
            P[0][i][i+1] = 1-P[0][i][0]
            perc += inc
        elif i < 99:
            P[0][i][0] = inc
            P[0][i][i+1] = 1-inc
        else:
            P[0][i][0] = inc
            P[0][i][i] = 1-inc
    return P

def get_rewards(env, holes=[], hole_val = 0):
    R = np.zeros(shape=(env.action_space.n, env.observation_space.n, env.observation_space.n))
    
    for a in range(env.action_space.n):
        for s in range(env.observation_space.n):
            for p, s_prime, r, done in env.P[s][a]:
                if s_prime in holes:
                    R[a,s,s_prime] = hole_val
                else:
                    R[a,s,s_prime] = r
    
    return R

def get_rewards_forest(R, r):
    for i in range(len(R)):
        if i == 0:
            pass
        elif i == 99:
            pass
        else:
            R[i][1]=r
    return R

def get_holes(env):
    holes = []
    for i in range(len(env.desc.ravel())):
        if env.desc.ravel()[i] == b'H':
            holes.append(i)
    return holes

def plot_val_pol_comp(val, pol, discounts, xlabel, ylabel, series_label, series_label_2, title, experiment, output_folder):
    plt.plot(val, label=series_label)
    plt.plot(pol, label=series_label_2)
    plt.xticks(ticks=list(range(len(discounts))), labels=discounts)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_folder+'/'+experiment+'_value_v_policy_comp.png')
    plt.close()
    plt.figure()
    
def plot_val_pol_match(val, pol, discounts, title, experiment, output_folder):
    '''
    The code for this table creation was inspired by @empet at this url: https://chart-studio.plotly.com/~empet/14689/table-with-cells-colored-according-to-th/#/
    '''
    matches = [str(pol[i] == val[i]) for i in range(len(discounts))]
    trace = dict(type = 'table',\
             columnwidth= [15] + [10],\
             header = dict(height = 50,\
                           values = [['<b>Discount</b>'], ['<b>Policy Match</b>']],\
                           line = dict(color='rgb(50, 50, 50)'),\
                           align = 'left', \
                           font = dict(color=['rgb(45, 45, 45)'], size=14),\
                           fill = dict( color = 'rgb(235, 235, 235)' )\
                              ),\
             cells = dict(values = [discounts, matches],\
                          line = dict(color='#506784'),\
                          align = 'left',\
                          font = dict(color=['black'], size=15),\
                          height = 30,\
                          fill = dict(color=['rgb(245, 245, 245)',\
                                            ['rgba(0, 250, 0, 0.8)' if x == 'True' else 'rgba(250, 0, 0, 0.8)' for x in matches] ])))    
    
    layout = dict(width=400, height=505, autosize=True, showlegend=False, margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
    fig = go.Figure(data=[trace], layout=layout)
    fig.write_image(output_folder+'/'+experiment+'_value_v_policy_match.png')

def policy_evaluation(P, R, policy, actions, states, episodes = 1000, start_state=None, goal_state=None, death_states=[], goal_action=None):
    np.random.seed(13)
    tot_rewards = []
    tot_iters = []
    if states >= 100:
        limit = states*2
    else:
        limit = states**2
    
    while episodes > 0:
        if start_state == None:
            s = np.random.choice(states)
        else:
            s = start_state
        done = False
        # Create a variable to prevent from getting lost and having an infinite episode
        i = 1
        # Create a variable to track how many rewards were earned in an episode
        rewards = 0

        while not done:
            
            # Choose an action based on the policy
            a = policy[s]
            
            # Take action a and get back a new state s_prime and a reward r
            s_prime = np.random.choice(states, p=P[a][s])
            try:
                r = R[a][s][s_prime]
            except:
                r = R[s][a]
            
            # add r to rewards earned during episode
            rewards += r
            
            # Update s to be s_prime
            s = s_prime
            
            # iterate i
            i += 1
            
            # Check if we've finished the episode
            if s == goal_state:
                done = True
                episodes = episodes-1
                tot_rewards.append(rewards)
                tot_iters.append(i)
            elif s in death_states:
                done = True
                episodes = episodes-1
                tot_rewards.append(rewards)
                tot_iters.append(i)
            elif i >= limit:
                done = True
                episodes = episodes-1
                tot_rewards.append(rewards)
                tot_iters.append(i)
                
    return tot_rewards, tot_iters
    
    
def init_q(states, actions, meth='zero'):
    if meth == 'zero':
        Q = np.zeros((states, actions))
    elif meth == 'random':
        Q = np.random.rand((states, actions))
    else:
        raise TypeError('Allowable values for meth are zero or random')
    return Q

def q_learning(P, R, Q, actions, states, episodes = 1000, episode_limit = 100000, min_episode_limit = 1000, alpha=0.95, gamma=0.99, epsilon=1, min_epsilon = 0.01, decay = 0.99, start_state=None, goal_state=None, death_states=[], goal_action=None):
    
    # Create Lists to store episode metrics
    ep_rew = []
    ep_iter = []
    act_made = []
    states_visited = []
    
    # Run all episodes
    while episodes > 0:
        if start_state == None:
            valid_start = False
            while not valid_start:
                s = np.random.choice(states)
                if s not in death_states:
                    valid_start = True            
        else:
            s = start_state
        done = False
        # Create a variable to prevent from getting lost and having an infinite episode
        i = 1
        # Create a variable to track how many rewards were earned in an episode
        rewards = 0
        # Update episode limit based on exploration factor
        limit = max(episode_limit*epsilon, min_episode_limit)
#        limit = np.inf
        while not done:
            
            # Choose an epsilon greedy action
            if random.uniform(0, 1) < epsilon:
                a = np.random.randint(0,actions) # choose a random action
            else:
                a = np.argmax(Q[s])
            
            # Take action a and get back a new state s_prime and a reward r
            s_prime = np.random.choice(states, p=P[a][s])
            try:
                r = R[a][s][s_prime]
            except:
                r = R[s][a]
            
            act_made.append(a)
            states_visited.append(s_prime)
            # add r to rewards earned during episode
            rewards += r
            
            # Update Q Table 
            Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s_prime, :]) - Q[s, a])
            
            # Decay the value of epsilon
            epsilon = max(epsilon*decay, min_epsilon)
            
            # Update s to be s_prime
            s = s_prime
            
            # iterate i
            i += 1
            
            # Check if we've finished the episode
            if s == goal_state:
                done = True
                episodes = episodes-1
                ep_rew.append(rewards)
                ep_iter.append(i)
            elif s in death_states:
                done = True
                episodes = episodes-1
                ep_rew.append(rewards)
                ep_iter.append(i)
            elif a == goal_action:
                done = True
                episodes = episodes-1
                ep_rew.append(rewards)
                ep_iter.append(i)
            elif i >= limit:
                done = True
                episodes = episodes-1
                ep_rew.append(rewards)
                ep_iter.append(i)
                
    return Q, ep_rew, ep_iter, act_made, states_visited, epsilon
    
def test_learner(Q, P, R, actions, states, start_state=None, goal_state=None, death_states=[], goal_action=None):
    if start_state == None:
        if start_state == None:
            valid_start = False
            while not valid_start:
                s = np.random.choice(states)
                if s not in death_states:
                    valid_start = True         
    else:
        s = start_state
        
    done = False
    won = False
    died = False

    # Create a variable to track how many rewards were earned in an episode
    rewards = 0

    act_made = []
    states_visited = []
    
    i = 1
    allowance = min(states**2, max(states*2, 2000))
    
    
    while not done:
        
        # Choose an action based on Q table
        a = np.argmax(Q[s])
        
        # Take action a and get back a new state s_prime and a reward r
        s_prime = np.random.choice(states, p=P[a][s])
        try:
            r = R[a][s][s_prime]
        except:
            r = R[s][a]
        
        act_made.append(a)
        states_visited.append(s_prime)
        # add r to rewards earned during episode
        rewards += r
        
        # Update s to be s_prime
        s = s_prime
        
        # Iterate i
        i += 1
        
        # Check if we've finished the episode
        if s == goal_state:
            done = True
            won = True
        elif a == goal_action:
            done = True
        elif s in death_states:
            done = True
            died = True
        elif i >= allowance:
            done = True

                
    return act_made, states_visited, won, rewards, died

def Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha=0.95, gamma=0.99, min_epsilon=0.01, goal_action=None):
    np.random.seed(13)
    wins = []
    deaths = []
    actions_taken = []
    actions_taken_win = []
    actions_taken_loss = []
    tot_rew = []
    
    for e in eps:
        Q_star, rewards, q_iters, act_made, states_visited, epsilon = q_learning(P, R, Q, actions, states, episodes = e, alpha=alpha, gamma=gamma, epsilon = 1, start_state=start_state, goal_state=goal_state, death_states=holes, goal_action=goal_action)
        
        # Test Q Learner
        win_count = 0
        death_count = 0
        actions_needed = 0
        actions_needed_win = 0
        actions_needed_loss = 0
        rewards_earned = []
        for i in range(100):
            act_made, states_visited, won, rewards, died = test_learner(Q_star, P, R, actions, states, start_state, goal_state, holes, goal_action=goal_action)
            win_count += int(won)
            death_count += int(died)
            actions_needed += len(act_made)
            rewards_earned.append(rewards)
            if won:
                actions_needed_win += len(act_made)
            elif died:
                actions_needed_loss += len(act_made)
        wins.append(win_count/100)
        deaths.append(death_count/100)
        actions_taken.append(actions_needed/100)
        tot_rew.append(np.mean(rewards_earned))
        if win_count > 0:
            actions_taken_win.append(actions_needed_win/win_count)
        else:
            actions_taken_win.append(0)
        if death_count > 0:
            actions_taken_loss.append(actions_needed_loss/death_count)
        else:
            actions_taken_loss.append(0)
    
    return wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew

def Q_wins(wins, episodes, xlabel, ylabel, title, experiment, output_folder):
    plt.plot(wins)
    plt.xticks(ticks=list(range(len(episodes))), labels=episodes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_folder+'/'+experiment+'_Q_learner_win_rate.png', bbox_inches='tight')
    plt.close()
    plt.figure()
    
def Q_losses(deaths, episodes, xlabel, ylabel, title, xlabel2, ylabel2, title2, experiment, output_folder):
    plt.plot(deaths)
    plt.xticks(ticks=list(range(len(episodes))), labels=episodes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_folder+'/'+experiment+'_Q_learner_death_rate.png')
    plt.close()
    plt.figure()

def Q_stuck(wins, deaths, episodes, xlabel, ylabel, title, experiment, output_folder): 
    stuck = [max(1-deaths[x]-wins[x],0) for x in range(len(deaths))]
    plt.plot(stuck)
    plt.xticks(ticks=list(range(len(episodes))), labels=episodes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_folder+'/'+experiment+'_Q_learner_stuck_rate.png')
    plt.close()
    plt.figure()

def Q_rewards(tot_rew, episodes, xlabel, ylabel, title, experiment, output_folder):
    plt.plot(tot_rew)
    plt.xticks(ticks=list(range(len(episodes))), labels=episodes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_folder+'/'+experiment+'_Q_learner_average_rewards_earned.png')
    plt.close()
    plt.figure()
    
def Q_stacked(wins, deaths, stuck, params, xlabel, ylabel, title, experiment, output_folder):
    width = 0.35
    fig, ax = plt.subplots()
    
    ax.bar(params, wins, width, label='Wins')
    ax.bar(params, stuck, width, bottom=wins, label='Stuck')
    ax.bar(params, deaths, width, bottom=stuck, label='Deaths')
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    plt.savefig(output_folder+'/'+experiment+'_Q_learner_success.png')
    plt.close()
    plt.figure()