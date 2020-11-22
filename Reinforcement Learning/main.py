import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import mdptoolbox
import mdptoolbox.example
import numpy as np
from util import get_transitions, get_transitions_forest, get_rewards, get_rewards_forest, get_holes, plot_val_pol_comp, policy_evaluation, init_q, q_learning, test_learner, Q_iters, Q_wins, Q_losses, Q_stuck, Q_rewards, plot_val_pol_match, Q_stacked

'''
Experiment 1 - 8x8 FrozenLake with +1 reward at the goal state and 0 everywhere else
'''

# Create environment for FrozenLake from OpenAI Gym
env = gym.make("FrozenLake-v0")
env.reset()
actions = env.action_space.n
states = env.observation_space.n
start_state = 0
goal_state = states-1

# Extract transition matrix and reward matrix
P = get_transitions(env)
R = get_rewards(env)

# Get Death States (i.e. Holes)
holes = get_holes(env)

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_1_iterations', 'frozen_lake')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_1_running_time', 'frozen_lake')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_1_policy_rewards', 'frozen_lake')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_1_policy_iters', 'frozen_lake')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_1_policy_match', 'frozen_lake')

# Train Q Learner
Q = init_q(states, actions, 'zero')

# Run Q learning 
rn = np.random.RandomState(13)
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_1_wins', 'frozen_lake')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_1_deaths', 'frozen_lake')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_1_rewards', 'frozen_lake')

'''
Experiment 2 - 8x8 FrozenLake with +1 reward at the goal state, -1 in all holes and 0 everywhere else
'''

# Create environment for FrozenLake from OpenAI Gym
env = gym.make("FrozenLake-v0")
env.reset()
actions = env.action_space.n
states = env.observation_space.n
start_state = 0
goal_state = states-1

# Get Death States (i.e. Holes)
holes = get_holes(env)

# Extract transition matrix and reward matrix
P = get_transitions(env)
R = get_rewards(env, holes = holes, hole_val = -1)

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_2_iterations', 'frozen_lake')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_2_policy_rewards', 'frozen_lake')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_2_policy_iters', 'frozen_lake')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_2_running_time', 'frozen_lake')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_2_policy_match', 'frozen_lake')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)

## Top 5 hyper-parameter values to graph
#top_5 = np.array(tot_wins).argsort()[-5:][::-1]
#tot_wins_2 = [tot_wins[x] for x in top_5]
#tot_deaths_2 = [tot_deaths[x] for x in top_5]
#tot_stuck_2 = [tot_stuck[x] for x in top_5]
#param_labels_2 = [param_labels[x] for x in top_5]
# Create Graphs
Q_wins(tot_wins, param_labels, 'Alpha-Gamma', 'Win Percentage', 'Q Learner Hyper-Parameter Tuning Results', 'exp_2_tuning', 'frozen_lake')
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_2_wins', 'frozen_lake')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_2_deaths', 'frozen_lake')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_2_stuck', 'frozen_lake')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_2_rewards', 'frozen_lake')

'''
Experiment 3 - 8x8 FrozenLake with +1 reward at the goal state, -0.2 in all holes and 0 everywhere else
'''

# Create environment for FrozenLake from OpenAI Gym
env = gym.make("FrozenLake-v0")
env.reset()
actions = env.action_space.n
states = env.observation_space.n
start_state = 0
goal_state = states-1

# Get Death States (i.e. Holes)
holes = get_holes(env)

# Extract transition matrix and reward matrix
P = get_transitions(env)
R = get_rewards(env, holes = holes, hole_val = -0.2)

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_3_iterations', 'frozen_lake')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_3_policy_rewards', 'frozen_lake')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_3_policy_iters', 'frozen_lake')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_3_running_time', 'frozen_lake')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_3_policy_match', 'frozen_lake')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_3_wins', 'frozen_lake')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_3_deaths', 'frozen_lake')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_3_stuck', 'frozen_lake')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_3_rewards', 'frozen_lake')

'''
Experiment 4 - 20x20 FrozenLake with +1 reward at the goal state, -0.2 in all holes and o everywhere else
'''

big_map = generate_random_map(size=20, p=0.8)

env = gym.make("FrozenLake-v0", desc=big_map)
env.reset()
actions = env.action_space.n
states = env.observation_space.n
start_state = 0
goal_state = states-1

# Get Death States (i.e. Holes)
holes = get_holes(env)

# Extract transition matrix and reward matrix
P = get_transitions(env)
R = get_rewards(env, holes = holes, hole_val = -0.2)

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_4_iterations', 'frozen_lake')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_4_policy_rewards', 'frozen_lake')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_4_policy_iters', 'frozen_lake')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_4_running_time', 'frozen_lake')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_4_policy_match', 'frozen_lake')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_4_wins', 'frozen_lake')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_4_deaths', 'frozen_lake')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_4_stuck', 'frozen_lake')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_4_rewards', 'frozen_lake')

# Rerun Q Learning without a start state - let the Q Learner randomly restart and train over greater portion of state space
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, None, goal_state, holes, eps, epsilon)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_4_wins_random_Q', 'frozen_lake')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_4_deaths_random_Q', 'frozen_lake')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_4_stuck_random_Q', 'frozen_lake')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_4_rewards_random_Q', 'frozen_lake')

# Rerun Q Learning without a start state and with a min epsilon of 0.5
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, None, goal_state, holes, eps, epsilon, min_epsilon=0.5)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_4_wins_more_exploration', 'frozen_lake')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_4_deaths_more_exploration', 'frozen_lake')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_4_stuck_more_exploration', 'frozen_lake')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_4_rewards_more_exploration', 'frozen_lake')


'''
Experiment 5 - 50x50 FrozenLake with +1 reward at the goal state, -0.2 in all holes and o everywhere else
'''

big_map = generate_random_map(size=50, p=0.8)

env = gym.make("FrozenLake-v0", desc=big_map)
env.reset()
actions = env.action_space.n
states = env.observation_space.n
start_state = None
goal_state = states-1

# Get Death States (i.e. Holes)
holes = get_holes(env)

# Extract transition matrix and reward matrix
P = get_transitions(env)
R = get_rewards(env, holes = holes, hole_val = -0.02)

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_5_iterations', 'frozen_lake')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_5_policy_rewards', 'frozen_lake')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_5_policy_iters', 'frozen_lake')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_5_running_time', 'frozen_lake')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_5_policy_match', 'frozen_lake')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000, 10000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, min_epsilon=0.5)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_5_wins', 'frozen_lake')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_5_deaths', 'frozen_lake')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_5_stuck', 'frozen_lake')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_5_rewards', 'frozen_lake')

''' 
Experiment 6 - 3 year Forest Management with 10% probability of fire
'''

P, R = mdptoolbox.example.forest()
start_state = 0
holes = [0]

actions = P.shape[0]
states = P.shape[1]

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_6_iterations', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_6_policy_rewards', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_6_policy_iters', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_6_running_time', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_6_policy_match', 'forest_management')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_6_wins', 'forest_management')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_6_deaths', 'forest_management')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_6_stuck', 'forest_management')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_6_rewards', 'forest_management')

''' 
Experiment 7 - 3 year Forest Management with 25% probability of fire
'''

P, R = mdptoolbox.example.forest(p=0.25)
start_state = 0
holes = [0]

actions = P.shape[0]
states = P.shape[1]

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_7_iterations', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_7_policy_rewards', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_7_policy_iters', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_7_running_time', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_7_policy_match', 'forest_management')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_7_wins', 'forest_management')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_7_deaths', 'forest_management')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_7_stuck', 'forest_management')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_7_rewards', 'forest_management')

''' 
Experiment 8 - 3 year Forest Management with 50% probability of fire
'''

P, R = mdptoolbox.example.forest(p=0.50)
start_state = 0
holes = [0]

actions = P.shape[0]
states = P.shape[1]

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_8_iterations', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_8_policy_rewards', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_8_policy_iters', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_8_running_time', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_8_policy_match', 'forest_management')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_8_wins', 'forest_management')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_8_deaths', 'forest_management')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_8_stuck', 'forest_management')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_8_rewards', 'forest_management')

''' 
Experiment 9 - 3 year Forest Management with 75% probability of fire
'''

P, R = mdptoolbox.example.forest(p=0.75)
start_state = 0
holes = [0]

actions = P.shape[0]
states = P.shape[1]

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_9_iterations', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_9_policy_rewards', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_9_policy_iters', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_9_running_time', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_9_policy_match', 'forest_management')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_9_wins', 'forest_management')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_9_deaths', 'forest_management')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_9_stuck', 'forest_management')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_9_rewards', 'forest_management')

''' 
Experiment 10 - 3 year Forest Management with 100% probability of fire
'''

P, R = mdptoolbox.example.forest(p=0.99)
start_state = 0
holes = [0]

actions = P.shape[0]
states = P.shape[1]

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=goal_state, death_states=holes)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_10_iterations', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_10_policy_rewards', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_10_policy_iters', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_10_running_time', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_10_policy_match', 'forest_management')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, goal_state, holes, eps, epsilon, alpha, gamma)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_10_wins', 'forest_management')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_10_deaths', 'forest_management')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_10_stuck', 'forest_management')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_10_rewards', 'forest_management')

''' 
Experiment 11 - 100 year Forest Management with 10% probability of fire
'''

P, R = mdptoolbox.example.forest(S=100)
start_state = 0
holes = [0]

actions = P.shape[0]
states = P.shape[1]

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_11_iterations', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_11_policy_rewards', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_11_policy_iters', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_11_running_time', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_11_policy_match', 'forest_management')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, None, holes, eps, epsilon, alpha, gamma, goal_action=1)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, None, holes, eps, epsilon, alpha, gamma, goal_action=1)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_11_wins', 'forest_management')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_11_deaths', 'forest_management')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_11_stuck', 'forest_management')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_11_rewards', 'forest_management')

# Re-Run Q Learning with more exploration and no start state
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, None, goal_state, holes, eps, epsilon, alpha, gamma, goal_action=1, min_epsilon = 0.5)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_11_wins_2', 'forest_management')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_11_deaths_2', 'forest_management')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_11_stuck_2', 'forest_management')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_11_rewards_2', 'forest_management')

''' 
Experiment 12 - 100 year Forest Management with 0.5% chance of fire 
'''

P, R = mdptoolbox.example.forest(S=100, p=0.005)
start_state = 0
holes = [0]

actions = P.shape[0]
states = P.shape[1]

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_12_iterations', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_12_policy_rewards', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_12_policy_iters', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_12_running_time', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_12_policy_match', 'forest_management')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, None, holes, eps, epsilon, alpha, gamma, goal_action=1, min_epsilon=0.5)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, None, holes, eps, epsilon, alpha, gamma, goal_action=1)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_12_wins', 'forest_management')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_12_deaths', 'forest_management')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_12_stuck', 'forest_management')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_12_rewards', 'forest_management')


''' 
Experiment 13 - Optimizing to wait 100 years
'''

P, R = mdptoolbox.example.forest(S=100, p=0.0005, r1=100)
start_state = 0
holes = [0]

actions = P.shape[0]
states = P.shape[1]

# Run value iteration for different rewards for cutting
# We know 1 is too enticing to cause the agent to not wait
rewards = [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, -1]
d = 0.9
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for r in rewards:
    P, R = mdptoolbox.example.forest(S=100, p=0.005)
    R = get_rewards_forest(R, r)
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, rewards, 'Cutting Reward', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_13_iterations_1', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, rewards, 'Cutting Reward', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_13_policy_rewards_1', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, rewards, 'Cutting Reward', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_13_policy_iters_1', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, rewards, 'Cutting Reward', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_13_running_time_1', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, rewards, 'Policy vs Value Iteration Policies', 'exp_13_policy_match_1', 'forest_management')

# reward = 0.0001 shows some unique behavior

# Run value iteration for different rewards for r1
# We know 4 isn't enticing enough to get the agent to wait for it
rewards = [100, 500, 1000, 10000, 25000, 50000, 100000]
d = 0.99
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for r in rewards:
    P, R = mdptoolbox.example.forest(S=100, p=0.005, r1=r)
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, rewards, 'Waiting Reward', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_13_iterations_2', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, rewards, 'Waiting Reward', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_13_policy_rewards_2', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, rewards, 'Waiting Reward', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_13_policy_iters_2', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, rewards, 'Waiting Reward', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_13_running_time_2', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, rewards, 'Policy vs Value Iteration Policies', 'exp_13_policy_match_1', 'forest_management')

# over 50,000 has an impact
# Run value iteration for different fire probabilities

probs = [0.000001, 0.0000001, 0.00000001, 0.00000000001]
d = 0.99
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for r in probs:
    P, R = mdptoolbox.example.forest(S=100, p=0)
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, probs, 'Fire Probability', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_13_iterations_3', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, probs, 'Fire Probability', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_13_policy_rewards_3', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, probs, 'Fire Probability', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_13_policy_iters_3', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, probs, 'Fire Probability', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_13_running_time_3', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, probs, 'Policy vs Value Iteration Policies', 'exp_13_policy_match_3', 'forest_management')

# None make much of a difference

P, R = mdptoolbox.example.forest(S=100, p=0.005, r1=50000)
R = get_rewards_forest(R, 0.0001)
start_state = 0
holes = [0]

actions = P.shape[0]
states = P.shape[1]

# Run value iteration for different discount factors
discounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
val_iters = []
val_policies = []
val_values = []
val_run_times = []
val_rewards = []
val_run_iters = []

pol_iters = []
pol_policies = []
pol_values = []
pol_run_times = []
pol_rewards = []
pol_run_iters = []
for d in discounts:
    vi = mdptoolbox.mdp.ValueIteration(P, R, d)
    vi.setSilent()
    vi.run()
    val_run_times.append(vi.time)
    val_policies.append(vi.policy)
    val_iters.append(vi.iter)
    val_values.append(vi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, vi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    val_rewards.append(np.mean(tot_rewards))
    val_run_iters.append(np.mean(tot_iters))
    
    pi = mdptoolbox.mdp.PolicyIteration(P, R, d)
    pi.setSilent()
    pi.run()
    pol_run_times.append(pi.time)
    pol_policies.append(pi.policy)
    pol_iters.append(pi.iter)
    pol_values.append(pi.V)
    tot_rewards, tot_iters = policy_evaluation(P, R, pi.policy, actions, states, start_state=start_state, goal_state=None, death_states=holes, goal_action=1)
    pol_rewards.append(np.mean(tot_rewards))
    pol_run_iters.append(np.mean(tot_iters))

plot_val_pol_comp(val_iters, pol_iters, discounts, 'Discount Rate', 'Iterations', 'Value Iteration', 'Policy Iteration', 'Iterations to Convergence for Various Discount Factors', 'exp_13_iterations', 'forest_management')
plot_val_pol_comp(val_rewards, pol_rewards, discounts, 'Discount Rate', 'Average Rewards Earned', 'Value Iteration', 'Policy Iteration', 'Average Reward Earned Following Optimal Policy', 'exp_13_policy_rewards', 'forest_management')
plot_val_pol_comp(val_run_iters, pol_run_iters, discounts, 'Discount Rate', 'Average Iterations Needed', 'Value Iteration', 'Policy Iteration', 'Average Iterations Needed Following Optimal Policy', 'exp_13_policy_iters', 'forest_management')
plot_val_pol_comp(val_run_times, pol_run_times, discounts, 'Discount Rate', 'Running Time', 'Value Iteration', 'Policy Iteration', 'Running Time to Convergence for Various Discount Factors', 'exp_13_running_time', 'forest_management')
plot_val_pol_match(val_policies, pol_policies, discounts, 'Policy vs Value Iteration Policies', 'exp_13_policy_match', 'forest_management')

# Tune Hyper-parameters for Q Learning
params = [(0.99, 0.99), (0.99, 0.9), (0.99, 0.75), (0.99, 0.5), (0.99, 0.25), (0.99, 0.1), (0.95, 0.99), (0.95, 0.9), (0.95, 0.75), (0.95, 0.5), (0.95, 0.25), (0.95, 0.1), (0.9, 0.99), (0.9, 0.9), (0.9, 0.75), (0.9, 0.5), (0.9, 0.25), (0.9, 0.1), (0.75, 0.99), (0.75, 0.9), (0.75, 0.75), (0.75, 0.5), (0.75, 0.25), (0.75, 0.1), (0.5, 0.99), (0.5, 0.9), (0.5, 0.75), (0.5, 0.5), (0.5, 0.25), (0.5, 0.1), (0.25, 0.99), (0.25, 0.9), (0.25, 0.75), (0.25, 0.5), (0.25, 0.25), (0.25, 0.1), (0.1, 0.99), (0.1, 0.9), (0.1, 0.75), (0.1, 0.5), (0.1, 0.25), (0.1, 0.1)]
param_labels = ['0.99-0.99', '0.99-0.9', '0.99-0.75', '0.99-0.5', '0.99-0.25', '0.99-0.1', '0.95-0.99', '0.95-0.9', '0.95-0.75', '0.95-0.5', '0.95-0.25', '0.95-0.1', '0.9-0.99', '0.9-0.9', '0.9-0.75', '0.9-0.5', '0.9-0.25', '0.9-0.1', '0.75-0.99', '0.75-0.9', '0.75-0.75', '0.75-0.5', '0.75-0.25', '0.75-0.1', '0.5-0.99', '0.5-0.9', '0.5-0.75', '0.5-0.5', '0.5-0.25', '0.5-0.1', '0.25-0.99', '0.25-0.9', '0.25-0.75', '0.25-0.5', '0.25-0.25', '0.25-0.1', '0.1-0.99', '0.1-0.9', '0.1-0.75', '0.1-0.5', '0.1-0.25', '0.1-0.1']
tot_wins = []
tot_deaths = []
tot_stuck = []
tot_reward = []
for alpha, gamma in params:
    Q = init_q(states, actions, 'zero')
    eps = [200]
    epsilon = 1
    wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, None, holes, eps, epsilon, alpha, gamma, goal_action=1, min_epsilon=0.5)
    tot_wins.append(np.mean(wins))
    tot_deaths.append(np.mean(deaths))
    tot_stuck.append(max(1-np.mean(wins)-np.mean(deaths),0))

# Best values
i = np.argmax(tot_wins)
alpha, gamma = params[i]

# Run Q learning 
Q = init_q(states, actions, 'zero')
eps = [1, 10, 25, 75, 100, 250, 500, 1000]
epsilon = 1
wins, deaths, actions_taken, actions_taken_win, actions_taken_loss, tot_rew = Q_iters(P, R, Q, actions, states, start_state, None, holes, eps, epsilon, alpha, gamma, goal_action=1)

# Create Graphs
Q_wins(wins, eps, 'Episodes', 'Win Percentage', 'Q Learner Win Rate', 'exp_13_wins', 'forest_management')
Q_losses(deaths, eps, 'Episodes', 'Death Percentage', 'Q Learner Death Rate', 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_13_deaths', 'forest_management')
Q_stuck(wins, deaths, eps, 'Episodes', 'Stuck Percentage', 'Q Learner Stuck Rate', 'exp_13_stuck', 'forest_management')
Q_rewards(tot_rew, eps, 'Episodes', 'Average Rewards', 'Q Learner Avg Rewards', 'exp_13_rewards', 'forest_management')

