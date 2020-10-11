'''
NOTE: this code makes use of several packages (referenced below) but none of them need to be specifically modified to run this code assuming you install the conda environment in step 2


References:
Several python packages were used in this repository that I would like to acknowledge.
Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar. (2020, March 18). pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3). Zenodo. http://doi.org/10.5281/zenodo.3715232

MLROSE: Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and Search package for Python. https://github.com/gkhayes/mlrose. Accessed: 9/28/2020
MLROSE-HIIVE fork was also utilized: Rollings, A. (2020). mlrose: Machine Learning, Randomized Optimization and Search package for Python. https://pypi.org/project/mlrose-hiive/ Accessed: 9/28/2020

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

Matplotlib: A 2D graphics environment, Hunter J.D., Computing in Science & Engineering. PP 90-95, 2007. 

In addition data for this project was taken from Kaggle.  Special thanks to the following:
Shruti Lyyer (@shruti_lyyer). (4/3/2019). Churn modeling, version 1. Retrieved 9/5/2020 from https://www.kaggle.com/shrutimechlearn/churn-modelling

'''
import mlrose_hiive as mlr
import numpy as np
import matplotlib.pyplot as plt
import time

# list of random seeds needed for analysis
random_seeds = [13, 37209, 5753, 63727, 76817, 87323, 36696, 35310, 93699, 52266, 35012, 68469, 57337, 72890, 16204, 71870]

''' 
    Problem 1 - Four Peaks 
    Should highlight Genetic Algorithms strengths.
    Expect RHC and SA to get stuck in local optimum while GA finds global optimum
    
'''

fitness = mlr.FourPeaks()
problem = mlr.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)
problem.set_mimic_fast_mode(True)

# problems and data structures for fitted algorithm comparison
p1_fi, p2_fi, p3_fi, p4_fi = [],[],[],[]
p1_ff, p2_ff, p3_ff, p4_ff = [],[],[],[]
p1_ft, p2_ft, p3_ft, p4_ft = [],[],[],[]
problems = [mlr.DiscreteOpt(length = 10, fitness_fn = fitness, maximize = True, max_val =2), mlr.DiscreteOpt(length = 50, fitness_fn = fitness, maximize = True, max_val =2), mlr.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val =2), mlr.DiscreteOpt(length = 250, fitness_fn = fitness, maximize = True, max_val =2)]
problem_names = ['Four Peaks N=10', 'Four Peaks N=50', 'Four Peaks N=100', 'Four Peaks N=250']

# Random Hill Climbing

# Open a txt file to log data in
file = open("Output/RHC_P1_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    RHC has 2 main parameters that we can tune: max_restarts and max_attempts (which is our stopping criteria)
    We want to find the combination of these 2 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of random hill climbing for different values of max_restarts
file.write('Max Restart Analysis\n')
max_restarts = [1, 5, 10, 15, 20, 25]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for r in max_restarts:
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=r, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Restart Value: ' + str(r) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of SA for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Random Hill Climbing Fitness of Various \nRestart Values on Four Peaks Problem')
plt.legend(legend, max_restarts, title='Max Restarts')
plt.savefig('Output/P1_RHC_Max_Restarts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Max Restarts of 1 & 5 end up with the same value and 15, 20, 25 do as well 
# Let's try a few different values to see how that changes things
max_restarts = [1, 10, 15, 50]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for r in max_restarts:
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=r, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Restart Value: ' + str(r) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max restart value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Random Hill Climbing Fitness of Various \nRestart Values on Four Peaks Problem')
plt.legend(legend, max_restarts, title='Max Restarts')
plt.savefig('Output/P1_RHC_Max_Restarts_fitness_2.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect RHC on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=a, max_iters=np.inf, restarts=1, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Random Hill Climbing Fitness of Various \nRestart Values on Four Peaks Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P1_RHC_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
max_attempts = [5, 25, 50, 75, 100]
max_restarts = [1, 10, 15, 50]
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for r in max_restarts:
    file.write('Max Restarts: ' + str(a) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for a in max_attempts:
        best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=a, max_iters=np.inf, restarts=r, init_state=None, curve=True, random_state=random_seeds[0])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_iterations.append(len(fitness_curve))
        iter_curves.append(fitness_curve)
        file.write('\tMax Attempts: ' + str(r) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
   ln, = plt.plot(y_vals)
   legend.append(ln)
plt.xticks(list(range(len(max_attempts))),max_attempts)
plt.ylabel('Best Fitness')
plt.xlabel('Max Attempts')
plt.title('Random Hill Climbing Fitness Comparison for \nHyper-Parameters on Four Peaks Problem')
plt.legend(legend, max_restarts, title='Max Restarts')
plt.savefig('Output/P1_RHC_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for RHC on this problem was found with max_attempts = 100, max_restarts = 50
    Taking these hyper-parameters we will run RHC on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.random_hill_climb(p, max_attempts=100, max_iters=np.inf, restarts=50, init_state=None, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Random Hill Climbing Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P1_RHC_Best_Fitness.png', bbox_inches='tight')
plt.close()


# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Random Hill Climbing on \nFour Peaks Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P1_RHC_Convergence.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Random Hill Climbing Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P1_RHC_Wall_Time.png', bbox_inches='tight')
plt.close()


# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])


file.close()

# Simulated Annealing

# Open a txt file to log data in
file = open("Output/SA_P1_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    SA has 2 main parameters that we can tune: schedule (which is our decay schedule) and max_attempts (which is our stopping criteria)
    We want to find the combination of these 2 hyper-parameters that gives us a best solution to this problem
'''
# First let's explore the performance of different decay schedules on the default SA algorithm from MLRose
schedules_geom = [mlr.GeomDecay(), mlr.GeomDecay(min_temp=0.5), mlr.GeomDecay(decay=0.99999), mlr.GeomDecay(decay=0.99999,min_temp=0.5)]
schedule_names_geom = ['Geom1', 'Geom2', 'Geom3', 'Geom4']
fitness_curves_geom = [] # store fitness curve for each decay function
best_fitnesses_geom = [] # Store best fitness for each function

schedules_arith = [mlr.ArithDecay(), mlr.ArithDecay(min_temp=0.5), mlr.ArithDecay(decay=0.0000001), mlr.ArithDecay(decay=0.0000001,min_temp=0.5)]
schedule_names_arith = ['Arith1', 'Arith2', 'Arith3', 'Arith4']
fitness_curves_arith = [] # store fitness curve for each decay function
best_fitnesses_arith = [] # Store best fitness for each function

schedules_exp = [mlr.ExpDecay(), mlr.ExpDecay(min_temp=0.5), mlr.ExpDecay(exp_const=0.00005), mlr.ExpDecay(exp_const=0.00005,min_temp=0.5)]
schedule_names_exp = ['Exp1', 'Exp2', 'Exp3', 'Exp4']
fitness_curves_exp = [] # store fitness curve for each decay function
best_fitnesses_exp = [] # Store best fitness for each function

for i in range(len(schedules_geom)):
    schedule = schedules_geom[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_geom[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_geom.append(fitness_curve)
    best_fitnesses_geom.append(best_fitness)
for i in range(len(schedules_arith)):
    schedule = schedules_arith[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_arith[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_arith.append(fitness_curve)
    best_fitnesses_arith.append(best_fitness)
for i in range(len(schedules_exp)):
    schedule = schedules_exp[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_exp[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_exp.append(fitness_curve)
    best_fitnesses_exp.append(best_fitness)
# Plot curve showing performance of SA for each decay function
fig, axs = plt.subplots(2,2, figsize=(15,15))
fig.suptitle('Simulated Annealing Fitness of Various \nDecay Functions on Four Peaks Problem')
legend = []
for y_vals in fitness_curves_geom:
    ln, = axs[0,0].plot(y_vals)
    legend.append(ln)
axs[0,0].set_title('Geometric Decay')
axs[0,0].legend(legend, schedule_names_geom, title='Decay Function')
legend = []
for y_vals in fitness_curves_arith:
    ln, = axs[0,1].plot(y_vals)
    legend.append(ln)
axs[0,1].set_title('Arithmetic Decay')
axs[0,1].legend(legend, schedule_names_arith, title='Decay Function')
legend = []
for y_vals in fitness_curves_exp:
    ln, = axs[1,0].plot(y_vals)
    legend.append(ln)
axs[1,0].set_title('Exponential Decay')
axs[1,0].legend(legend, schedule_names_exp, title='Decay Function')
axs[1,1].axis('off')
for ax in axs.flat:
    ax.set(xlabel='Iteration', ylabel='Fitness')
for ax in axs.flat:
    ax.label_outer()
plt.savefig('Output/P1_SA_Decay_Function_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect SA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, max_attempts=a, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Simulated Annealing Fitness of Various \nRestart Values on Four Peaks Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P1_SA_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Finally let's see how different combinations of these hyperparameters work together
best_schedules = [mlr.GeomDecay(min_temp=0.5), mlr.ArithDecay(decay=0.0000001,min_temp=0.5), mlr.ExpDecay(min_temp=0.5)]
best_schedule_names = ['Geo2', 'Arith4', 'Exp2']
file.write('Hyper Parameter Combinations\n')
max_attempts = [5, 25, 50, 75, 100]
max_restarts = [1, 10, 15, 50]
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for i in range(len(best_schedules)):
    schedule = best_schedules[i]
    file.write('Schedule: ' + best_schedule_names[i] + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for a in max_attempts:
        best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=a, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[1])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_iterations.append(len(fitness_curve))
        iter_curves.append(fitness_curve)
        file.write('\tMax Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
   ln, = plt.plot(y_vals)
   legend.append(ln)
plt.xticks(list(range(len(max_attempts))),max_attempts)
plt.ylabel('Best Fitness')
plt.xlabel('Max Attempts')
plt.title('Simulated Annealing Fitness Comparison for \nHyper-Parameters on Four Peaks Problem')
plt.legend(legend, best_schedule_names, title='Schedule')
plt.savefig('Output/P1_SA_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

###### TODO: Consider the consequences of a constant temperature or changing the speed of decay or decay limit #####

'''
    Fitted Algorithm Runs

    The best performance for SA on this problem was found with max_attempts = 50 (it performed equally well with MA=100 so take the smaller value), schedule = Geo2
    Taking these hyper-parameters we will run SA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
schedule = best_schedules[0]
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.simulated_annealing(p, schedule, max_attempts=75, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[j+2])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Simulated Annealing Best Fitness over 10 Trials')
plt.tight_layout()
fig.savefig('Output/P1_SA_Best_Fitness.png', bbox_inches='tight')
##### TODO: Consider changing last graph of averages to average/max possible score ######

# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Simulated Annealing on \nFour Peaks Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
plt.tight_layout()
fig.savefig('Output/P1_SA_Convergence.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Simulated Annealing Wall Time over 10 Trials')
plt.tight_layout()
fig.savefig('Output/P1_SA_Wall_Time.png', bbox_inches='tight')
plt.close()

###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])


file.close()



# Genetic Algorithm

file = open("Output/GA_P1_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    GA has 3 main parameters that we can tune: pop_size, mutation_prob and max_attempts (which is our stopping criteria)
    We want to find the combination of these 3 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of random hill climbing for different values of max_restarts
file.write('Pop Size Analysis\n')
pop_sizes = [5, 25, 100, 200]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for s in pop_sizes:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=s, mutation_prob=0.1, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Population Size: ' + str(s) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of SA for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nPopulation Sizes on Four Peaks Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P1_GA_Pop_Size_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')


# Next we will experiment with different mutation probabilities
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for p in probs:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=200, mutation_prob=p, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Mutation Probability: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max restart value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nMutation Probabilities on Four Peaks Problem')
plt.legend(legend, probs, title='Mutation Probability')
plt.savefig('Output/P1_GA_Mutation_Probability_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect GA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=a, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of GA for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nMax Attempt Values on Four Peaks Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P1_GA_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

##### TODO Figure out a good visualization for 3 hyper-parameters #####
# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for s in pop_sizes:
    file.write('Population size: ' + str(s) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for p in probs:
        file.write('\tProbability: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
        best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=s, mutation_prob=p, max_attempts=25, max_iters=np.inf, curve=True, random_state=random_seeds[1])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_curves.append(len(fitness_curve))
        iter_iterations.append(fitness_curve)
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(probs))),probs)
plt.ylabel('Best Fitness')
plt.xlabel('Mutation Probability')
plt.title('Genetic Algorithm Fitness Comparison for \nHyper-Parameters on Four Peaks Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P1_GA_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for GA on this problem was found with max_attempts = 25, population = 200, mutation_probability = 0.3
    Taking these hyper-parameters we will run GA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.genetic_alg(p, pop_size=200, mutation_prob=0.4, max_attempts=100, max_iters=np.inf, curve=True, random_state=random_seeds[j+2])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Genetic Algorithm Best Fitness over 10 Trials')
plt.tight_layout()
fig.savefig('Output/P1_GA_Best_Fitness.png', bbox_inches='tight')
##### TODO: Consider changing last graph of averages to average/max possible score ######

# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Genetic Algorithm on \nFour Peaks Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
plt.tight_layout()
fig.savefig('Output/P1_GA_Convergence.png', bbox_inches='tight')
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Genetic Algorithm Wall Time over 10 Trials')
plt.tight_layout()
fig.savefig('Output/P1_GA_Wall_Time.png', bbox_inches='tight')
plt.close()

###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close()

# MIMIC

file = open("Output/MIMIC_P1_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    GA has 3 main parameters that we can tune: pop_size, keep_pct and max_attempts (which is our stopping criteria)
    We want to find the combination of these 3 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of MIMIC for different values of population size
file.write('Pop Size Analysis\n')
pop_sizes = [5, 25, 100, 200]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for s in pop_sizes:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=s, keep_pct=0.2, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Population Size: ' + str(s) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of MIMIC for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nPopulation Sizes on Four Peaks Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P1_MIMIC_Pop_Size_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')


# Next we will experiment with different keep_pct
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for p in probs:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=200, keep_pct=p, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Keep %: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max restart value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nKeep Percentages on Four Peaks Problem')
plt.legend(legend, probs, title='Keep %')
plt.savefig('Output/P1_MIMIC_Keep_Percentage_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect GA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=200, keep_pct=0.1, max_attempts=a, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of GA for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nMax Attempt Values on Four Peaks Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P1_MIMIC_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

##### TODO Figure out a good visualization for 3 hyper-parameters #####
# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for s in pop_sizes:
    file.write('Population size: ' + str(s) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for p in probs:
        best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=s, keep_pct=p, max_attempts=100, max_iters=np.inf, curve=True, random_state=random_seeds[0])      
        file.write('\tKeep Percentage: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_curves.append(len(fitness_curve))
        iter_iterations.append(fitness_curve)
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(probs))),probs)
plt.ylabel('Best Fitness')
plt.xlabel('Mutation Probability')
plt.title('Genetic Algorithm Fitness Comparison for \nHyper-Parameters on Four Peaks Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P1_MIMIC_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for GA on this problem was found with max_attempts = 25, population = 200, keep_pct = 0.2
    Taking these hyper-parameters we will run GA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.mimic(p, pop_size=200, keep_pct=0.2, max_attempts=25, max_iters=np.inf, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('MIMIC Best Fitness over 10 Trials')
plt.tight_layout()
fig.savefig('Output/P1_MIMIC_Best_Fitness.png', bbox_inches='tight')
##### TODO: Consider changing last graph of averages to average/max possible score ######

# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of MIMIC on \nFour Peaks Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
plt.tight_layout()
fig.savefig('Output/P1_MIMIC_Convergence.png', bbox_inches='tight')
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('MIMIC Wall Time over 10 Trials')
plt.tight_layout()
fig.savefig('Output/P1_MIMIC_Wall_Time.png', bbox_inches='tight')

# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close


# Graph Comparing Each Algorithm's Average performance on each of the 4 problem sizes
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_ff)), p1_ff, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_ff)), p2_ff, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_ff)), p3_ff, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_ff)), p4_ff, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Fitness for Each Algorithm on \nVarying Sizes of the Four Peaks Problem')
plt.tight_layout()
fig.savefig('Output/P1_ALL_Fitness.png', bbox_inches='tight')

# Graph comparing wall time for each of the algorithms
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_ft)), p1_ft, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_ft)), p2_ft, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_ft)), p3_ft, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_ft)), p4_ft, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Fitness for Each Algorithm on \nVarying Sizes of the Four Peaks Problem')
plt.tight_layout()
fig.savefig('Output/P1_ALL_Wall_Clock.png', bbox_inches='tight')

# Graph comparing iterations to converge for each of the algorithms
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_fi)), p1_fi, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_fi)), p2_fi, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_fi)), p3_fi, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_fi)), p4_fi, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Iterations to Convergence for Each Algorithm on \nVarying Sizes of the Four Peaks Problem')
plt.tight_layout()
fig.savefig('Output/P1_ALL_Iterations.png', bbox_inches='tight')

###### Problem 2 - One Max Problem #######

fitness = mlr.OneMax()
problem = mlr.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)
problem.set_mimic_fast_mode(True)

# problems and data structures for fitted algorithm comparison
p1_fi, p2_fi, p3_fi, p4_fi = [],[],[],[]
p1_ff, p2_ff, p3_ff, p4_ff = [],[],[],[]
p1_ft, p2_ft, p3_ft, p4_ft = [],[],[],[]
final_curves = []
final_fitnesses = []
final_times = []
problems = [mlr.DiscreteOpt(length = 10, fitness_fn = fitness, maximize = True, max_val =2), mlr.DiscreteOpt(length = 50, fitness_fn = fitness, maximize = True, max_val =2), mlr.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val =2), mlr.DiscreteOpt(length = 250, fitness_fn = fitness, maximize = True, max_val =2)]
problem_names = ['One Max N=10', 'One Max N=50', 'One Max N=100', 'One Max N=250']

# Random Hill Climbing

# Open a txt file to log data in
file = open("Output/RHC_P2_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    RHC has 2 main parameters that we can tune: max_restarts and max_attempts (which is our stopping criteria)
    We want to find the combination of these 2 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of random hill climbing for different values of max_restarts
file.write('Max Restart Analysis\n')
max_restarts = [1, 5, 10, 15, 20, 25]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for r in max_restarts:
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=r, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Restart Value: ' + str(r) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of SA for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Random Hill Climbing Fitness of Various \nRestart Values on One Max Problem')
plt.legend(legend, max_restarts, title='Max Restarts')
plt.savefig('Output/P2_RHC_Max_Restarts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect RHC on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=a, max_iters=np.inf, restarts=1, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Random Hill Climbing Fitness of Various \nRestart Values on One Max Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P2_RHC_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
max_attempts = [5, 25, 50, 75, 100]
max_restarts = [1, 10, 15, 50]
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for r in max_restarts:
    file.write('Max Restarts: ' + str(a) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for a in max_attempts:
        best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=a, max_iters=np.inf, restarts=r, init_state=None, curve=True, random_state=random_seeds[0])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_iterations.append(len(fitness_curve))
        iter_curves.append(fitness_curve)
        file.write('\tMax Attempts: ' + str(r) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
   ln, = plt.plot(y_vals)
   legend.append(ln)
plt.xticks(list(range(len(max_attempts))),max_attempts)
plt.ylabel('Best Fitness')
plt.xlabel('Max Attempts')
plt.title('Random Hill Climbing Fitness Comparison for \nHyper-Parameters on One Max Problem')
plt.legend(legend, max_restarts, title='Max Restarts')
plt.savefig('Output/P2_RHC_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for RHC on this problem was found with max_attempts = 50, max_restarts = 50
    Taking these hyper-parameters we will run RHC on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.random_hill_climb(p, max_attempts=50, max_iters=np.inf, restarts=50, init_state=None, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Random Hill Climbing Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P2_RHC_Best_Fitness.png', bbox_inches='tight')


# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Random Hill Climbing on \nOne Max Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P2_RHC_Convergence.png', bbox_inches='tight')
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Random Hill Climbing Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P2_RHC_Wall_Time.png', bbox_inches='tight')

###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close()
    
# Simulated Annealing

# Open a txt file to log data in
file = open("Output/SA_P2_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    SA has 2 main parameters that we can tune: schedule (which is our decay schedule) and max_attempts (which is our stopping criteria)
    We want to find the combination of these 2 hyper-parameters that gives us a best solution to this problem
'''
# First let's explore the performance of different decay schedules on the default SA algorithm from MLRose
schedules_geom = [mlr.GeomDecay(), mlr.GeomDecay(min_temp=0.5), mlr.GeomDecay(decay=0.99999), mlr.GeomDecay(decay=0.99999,min_temp=0.5)]
schedule_names_geom = ['Geom1', 'Geom2', 'Geom3', 'Geom4']
fitness_curves_geom = [] # store fitness curve for each decay function
best_fitnesses_geom = [] # Store best fitness for each function

schedules_arith = [mlr.ArithDecay(), mlr.ArithDecay(min_temp=0.5), mlr.ArithDecay(decay=0.0000001), mlr.ArithDecay(decay=0.0000001,min_temp=0.5)]
schedule_names_arith = ['Arith1', 'Arith2', 'Arith3', 'Arith4']
fitness_curves_arith = [] # store fitness curve for each decay function
best_fitnesses_arith = [] # Store best fitness for each function

schedules_exp = [mlr.ExpDecay(), mlr.ExpDecay(min_temp=0.5), mlr.ExpDecay(exp_const=0.00005), mlr.ExpDecay(exp_const=0.00005,min_temp=0.5)]
schedule_names_exp = ['Exp1', 'Exp2', 'Exp3', 'Exp4']
fitness_curves_exp = [] # store fitness curve for each decay function
best_fitnesses_exp = [] # Store best fitness for each function

for i in range(len(schedules_geom)):
    schedule = schedules_geom[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_geom[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_geom.append(fitness_curve)
    best_fitnesses_geom.append(best_fitness)
for i in range(len(schedules_arith)):
    schedule = schedules_arith[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_arith[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_arith.append(fitness_curve)
    best_fitnesses_arith.append(best_fitness)
for i in range(len(schedules_exp)):
    schedule = schedules_exp[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_exp[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_exp.append(fitness_curve)
    best_fitnesses_exp.append(best_fitness)
# Plot curve showing performance of SA for each decay function
fig, axs = plt.subplots(2,2, figsize=(15,15))
fig.suptitle('Simulated Annealing Fitness of Various \nDecay Functions on One Max Problem')
legend = []
for y_vals in fitness_curves_geom:
    ln, = axs[0,0].plot(y_vals)
    legend.append(ln)
axs[0,0].set_title('Geometric Decay')
axs[0,0].legend(legend, schedule_names_geom, title='Decay Function')
legend = []
for y_vals in fitness_curves_arith:
    ln, = axs[0,1].plot(y_vals)
    legend.append(ln)
axs[0,1].set_title('Arithmetic Decay')
axs[0,1].legend(legend, schedule_names_arith, title='Decay Function')
legend = []
for y_vals in fitness_curves_exp:
    ln, = axs[1,0].plot(y_vals)
    legend.append(ln)
axs[1,0].set_title('Exponential Decay')
axs[1,0].legend(legend, schedule_names_exp, title='Decay Function')
axs[1,1].axis('off')
for ax in axs.flat:
    ax.set(xlabel='Iteration', ylabel='Fitness')
for ax in axs.flat:
    ax.label_outer()
plt.savefig('Output/P2_SA_Decay_Function_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect SA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, max_attempts=a, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Simulated Annealing Fitness of Various \nRestart Values on One Max Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P2_SA_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Finally let's see how different combinations of these hyperparameters work together
best_schedules = [mlr.GeomDecay(min_temp=0.5), mlr.ExpDecay(), mlr.ExpDecay(min_temp=0.5)]
best_schedule_names = ['Geo2', 'Exp1', 'Exp2']
file.write('Hyper Parameter Combinations\n')
max_attempts = [5, 25, 50, 75, 100]
max_restarts = [1, 10, 15, 50]
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for i in range(len(best_schedules)):
    schedule = best_schedules[i]
    file.write('Schedule: ' + best_schedule_names[i] + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for a in max_attempts:
        best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=a, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[1])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_iterations.append(len(fitness_curve))
        iter_curves.append(fitness_curve)
        file.write('\tMax Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(max_attempts))),max_attempts)
plt.ylabel('Best Fitness')
plt.xlabel('Max Attempts')
plt.title('Simulated Annealing Fitness Comparison for \nHyper-Parameters on One Max Problem')
plt.legend(legend, best_schedule_names, title='Schedule')
plt.savefig('Output/P2_SA_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')


###### TODO: Consider the consequences of a constant temperature or changing the speed of decay or decay limit #####

'''
    Fitted Algorithm Runs

    The best performance for SA on this problem was found with max_attempts = 100, schedule = Exponential
    Taking these hyper-parameters we will run SA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
schedule = best_schedules[1]
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.simulated_annealing(p, schedule, max_attempts=100, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Simulated Annealing Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P2_SA_Best_Fitness.png', bbox_inches='tight')


# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Simulated Annealing on \nOne Max Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P2_SA_Convergence.png', bbox_inches='tight')
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Simulated Annealing Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P2_SA_Wall_Time.png', bbox_inches='tight')

###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close()
    
# Genetic Algorithm

file = open("Output/GA_P2_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    GA has 3 main parameters that we can tune: pop_size, mutation_prob and max_attempts (which is our stopping criteria)
    We want to find the combination of these 3 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of random hill climbing for different values of max_restarts
file.write('Pop Size Analysis\n')
pop_sizes = [5, 25, 100, 200]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for s in pop_sizes:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=s, mutation_prob=0.1, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Population Size: ' + str(s) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of SA for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nPopulation Sizes on One Max Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P2_GA_Pop_Size_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')


# Next we will experiment with different mutation probabilities
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for p in probs:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=200, mutation_prob=p, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Mutation Probability: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max restart value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nMutation Probabilities on One Max Problem')
plt.legend(legend, probs, title='Mutation Probability')
plt.savefig('Output/P2_GA_Mutation_Probability_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect GA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=a, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of GA for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nMax Attempt Values on One Max Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P2_GA_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

##### TODO Figure out a good visualization for 3 hyper-parameters #####
# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for s in pop_sizes:
    file.write('Population size: ' + str(s) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for p in probs:
        file.write('\tProbability: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
        best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=s, mutation_prob=p, max_attempts=25, max_iters=np.inf, curve=True, random_state=random_seeds[1])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_curves.append(len(fitness_curve))
        iter_iterations.append(fitness_curve)
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(probs))),probs)
plt.ylabel('Best Fitness')
plt.xlabel('Mutation Probability')
plt.title('Genetic Algorithm Fitness Comparison for \nHyper-Parameters on One Max Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P2_GA_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for GA on this problem was found with max_attempts = 100, population = 100, mutation_probability = 0.4
    Taking these hyper-parameters we will run GA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.genetic_alg(p, pop_size=100, mutation_prob=0.4, max_attempts=100, max_iters=np.inf, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Genetic Algorithm Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P2_GA_Best_Fitness.png', bbox_inches='tight')


# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Genetic Algorithm Climbing on \nOne Max Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P2_GA_Convergence.png', bbox_inches='tight')
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Genetic Algorithm Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P2_GA_Wall_Time.png', bbox_inches='tight')

###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close()

# MIMIC

file = open("Output/MIMIC_P2_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    GA has 3 main parameters that we can tune: pop_size, keep_pct and max_attempts (which is our stopping criteria)
    We want to find the combination of these 3 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of MIMIC for different values of population size
file.write('Pop Size Analysis\n')
pop_sizes = [5, 25, 100, 200]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for s in pop_sizes:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=s, keep_pct=0.2, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Population Size: ' + str(s) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of MIMIC for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nPopulation Sizes on One Max Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P2_MIMIC_Pop_Size_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')


# Next we will experiment with different keep_pct
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for p in probs:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=200, keep_pct=p, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Keep %: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max restart value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nKeep Percentages on One Max Problem')
plt.legend(legend, probs, title='Keep %')
plt.savefig('Output/P2_MIMIC_Keep_Percentage_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect GA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=200, keep_pct=0.1, max_attempts=a, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of GA for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nMax Attempt Values on One Max Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P2_MIMIC_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

##### TODO Figure out a good visualization for 3 hyper-parameters #####
# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for s in pop_sizes:
    file.write('Population size: ' + str(s) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for p in probs:
        best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=s, keep_pct=p, max_attempts=100, max_iters=np.inf, curve=True, random_state=random_seeds[0])      
        file.write('\tKeep Percentage: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_curves.append(len(fitness_curve))
        iter_iterations.append(fitness_curve)
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(probs))),probs)
plt.ylabel('Best Fitness')
plt.xlabel('Mutation Probability')
plt.title('Genetic Algorithm Fitness Comparison for \nHyper-Parameters on One Max Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P2_MIMIC_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for GA on this problem was found with max_attempts = 50, population = 200, keep_pct = 0.2
    Taking these hyper-parameters we will run GA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.mimic(p, pop_size=200, keep_pct=0.2, max_attempts=25, max_iters=np.inf, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('MIMIC Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P2_MIMIC_Best_Fitness.png', bbox_inches='tight')


# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of MIMIC on \nOne Max Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P2_MIMIC_Convergence.png', bbox_inches='tight')
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('MIMIC Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P2_MIMIC_Wall_Time.png', bbox_inches='tight')

###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close

# Graph Comparing Each Algorithm's Average performance on each of the 4 problem sizes
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_ff)), p1_ff, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_ff)), p2_ff, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_ff)), p3_ff, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_ff)), p4_ff, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Fitness for Each Algorithm on \nVarying Sizes of the One Max Problem')
plt.tight_layout()
fig.savefig('Output/P2_ALL_Fitness.png', bbox_inches='tight')

# Graph comparing wall time for each of the algorithms
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
plt.setp(axs, xscale='log')
axs[0,0].barh(np.arange(len(p1_ft)), p1_ft, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_ft)), p2_ft, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_ft)), p3_ft, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_ft)), p4_ft, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Wall Clock Time for Each Algorithm on \nVarying Sizes of the One Max Problem')
plt.tight_layout()
fig.savefig('Output/P2_ALL_Wall_Clock.png', bbox_inches='tight')

# Graph comparing iterations to converge for each of the algorithms
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_fi)), p1_fi, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_fi)), p2_fi, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_fi)), p3_fi, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_fi)), p4_fi, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Iterations to Convergence for Each Algorithm on \nVarying Sizes of the One Max Problem')
plt.tight_layout()
fig.savefig('Output/P2_ALL_Iterations.png', bbox_inches='tight')


""" ###### Problem 3 - FlipFlop ######

fitness = mlr.FlipFlop()
problem = mlr.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val = 2)
problem.set_mimic_fast_mode(True)

# problems and data structures for fitted algorithm comparison
p1_fi, p2_fi, p3_fi, p4_fi = [],[],[],[]
p1_ff, p2_ff, p3_ff, p4_ff = [],[],[],[]
p1_ft, p2_ft, p3_ft, p4_ft = [],[],[],[]
final_curves = []
final_fitnesses = []
final_times = []
# problems = [mlr.DiscreteOpt(length = 50, fitness_fn = fitness, maximize = True, max_val =2), mlr.DiscreteOpt(length = 100, fitness_fn = fitness, maximize = True, max_val =2), mlr.DiscreteOpt(length = 250, fitness_fn = fitness, maximize = True, max_val =2), mlr.DiscreteOpt(length = 1000, fitness_fn = fitness, maximize = True, max_val =2)]
# problem_names = ['Flip Flop N=50', 'Flip Flop N=100', 'Flip Flop N=250', 'Flip Flop N=1000']
problems = [mlr.DiscreteOpt(length = 250, fitness_fn = fitness, maximize = True, max_val =2)]
problem_names = ['Flip Flop N=250']
# Random Hill Climbing

# Open a txt file to log data in
file = open("Output/RHC_P3_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    RHC has 2 main parameters that we can tune: max_restarts and max_attempts (which is our stopping criteria)
    We want to find the combination of these 2 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of random hill climbing for different values of max_restarts
file.write('Max Restart Analysis\n')
max_restarts = [1, 5, 10, 15, 20, 25]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for r in max_restarts:
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=r, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Restart Value: ' + str(r) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of SA for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Random Hill Climbing Fitness of Various \nRestart Values on Flip Flop Problem')
plt.legend(legend, max_restarts, title='Max Restarts')
plt.savefig('Output/P3_RHC_Max_Restarts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect RHC on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=a, max_iters=np.inf, restarts=1, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Random Hill Climbing Fitness of Various \nRestart Values on Flip Flop Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P3_RHC_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
max_attempts = [5, 25, 50, 75, 100]
max_restarts = [1, 10, 15, 50]
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for r in max_restarts:
    file.write('Max Restarts: ' + str(a) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for a in max_attempts:
        best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem, max_attempts=a, max_iters=np.inf, restarts=r, init_state=None, curve=True, random_state=random_seeds[0])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_iterations.append(len(fitness_curve))
        iter_curves.append(fitness_curve)
        file.write('\tMax Attempts: ' + str(r) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(max_attempts))),max_attempts)
plt.ylabel('Best Fitness')
plt.xlabel('Max Attempts')
plt.title('Random Hill Climbing Fitness Comparison for \nHyper-Parameters on Flip Flop Problem')
plt.legend(legend, max_restarts, title='Max Restarts')
plt.savefig('Output/P3_RHC_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for RHC on this problem was found with max_attempts = 25, max_restarts = 50
    Taking these hyper-parameters we will run RHC on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.random_hill_climb(p, max_attempts=25, max_iters=np.inf, restarts=50, init_state=None, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Random Hill Climbing Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P3_RHC_Best_Fitness_1000.png', bbox_inches='tight')
plt.close()

# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Random Hill Climbing on \nFlip Flop Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P3_RHC_Convergence_1000.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Random Hill Climbing Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P3_RHC_Wall_Time_1000.png', bbox_inches='tight')
plt.close()
###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close()
    
# Simulated Annealing

# Open a txt file to log data in
file = open("Output/SA_P3_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    SA has 2 main parameters that we can tune: schedule (which is our decay schedule) and max_attempts (which is our stopping criteria)
    We want to find the combination of these 2 hyper-parameters that gives us a best solution to this problem
'''
# First let's explore the performance of different decay schedules on the default SA algorithm from MLRose
schedules_geom = [mlr.GeomDecay(), mlr.GeomDecay(min_temp=0.5), mlr.GeomDecay(decay=0.99999), mlr.GeomDecay(decay=0.99999,min_temp=0.5)]
schedule_names_geom = ['Geom1', 'Geom2', 'Geom3', 'Geom4']
fitness_curves_geom = [] # store fitness curve for each decay function
best_fitnesses_geom = [] # Store best fitness for each function

schedules_arith = [mlr.ArithDecay(), mlr.ArithDecay(min_temp=0.5), mlr.ArithDecay(decay=0.0000001), mlr.ArithDecay(decay=0.0000001,min_temp=0.5)]
schedule_names_arith = ['Arith1', 'Arith2', 'Arith3', 'Arith4']
fitness_curves_arith = [] # store fitness curve for each decay function
best_fitnesses_arith = [] # Store best fitness for each function

schedules_exp = [mlr.ExpDecay(), mlr.ExpDecay(min_temp=0.5), mlr.ExpDecay(exp_const=0.00005), mlr.ExpDecay(exp_const=0.00005,min_temp=0.5)]
schedule_names_exp = ['Exp1', 'Exp2', 'Exp3', 'Exp4']
fitness_curves_exp = [] # store fitness curve for each decay function
best_fitnesses_exp = [] # Store best fitness for each function

for i in range(len(schedules_geom)):
    schedule = schedules_geom[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_geom[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_geom.append(fitness_curve)
    best_fitnesses_geom.append(best_fitness)
for i in range(len(schedules_arith)):
    schedule = schedules_arith[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_arith[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_arith.append(fitness_curve)
    best_fitnesses_arith.append(best_fitness)
for i in range(len(schedules_exp)):
    schedule = schedules_exp[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_exp[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_exp.append(fitness_curve)
    best_fitnesses_exp.append(best_fitness)
# Plot curve showing performance of SA for each decay function
fig, axs = plt.subplots(2,2, figsize=(15,15))
fig.suptitle('Simulated Annealing Fitness of Various \nDecay Functions on Flip Flop Problem')
legend = []
for y_vals in fitness_curves_geom:
    ln, = axs[0,0].plot(y_vals)
    legend.append(ln)
axs[0,0].set_title('Geometric Decay')
axs[0,0].legend(legend, schedule_names_geom, title='Decay Function')
legend = []
for y_vals in fitness_curves_arith:
    ln, = axs[0,1].plot(y_vals)
    legend.append(ln)
axs[0,1].set_title('Arithmetic Decay')
axs[0,1].legend(legend, schedule_names_arith, title='Decay Function')
legend = []
for y_vals in fitness_curves_exp:
    ln, = axs[1,0].plot(y_vals)
    legend.append(ln)
axs[1,0].set_title('Exponential Decay')
axs[1,0].legend(legend, schedule_names_exp, title='Decay Function')
axs[1,1].axis('off')
for ax in axs.flat:
    ax.set(xlabel='Iteration', ylabel='Fitness')
for ax in axs.flat:
    ax.label_outer()
plt.savefig('Output/P3_SA_Decay_Function_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect SA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, max_attempts=a, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Simulated Annealing Fitness of Various \nRestart Values on Flip Flop Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P3_SA_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Finally let's see how different combinations of these hyperparameters work together
best_schedules = [mlr.GeomDecay(), mlr.GeomDecay(min_temp=0.5), mlr.ExpDecay(min_temp=0.5)]
best_schedule_names = ['Geo1', 'Geo2', 'Exp2']
file.write('Hyper Parameter Combinations\n')
max_attempts = [5, 25, 50, 75, 100]
max_restarts = [1, 10, 15, 50]
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for i in range(len(best_schedules)):
    schedule = best_schedules[i]
    file.write('Schedule: ' + best_schedule_names[i] + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for a in max_attempts:
        best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=a, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[1])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_iterations.append(len(fitness_curve))
        iter_curves.append(fitness_curve)
        file.write('\tMax Attempts: ' + str(r) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(max_attempts))),max_attempts)
plt.ylabel('Best Fitness')
plt.xlabel('Max Attempts')
plt.title('Simulated Annealing Fitness Comparison for \nHyper-Parameters on Flip Flop Problem')
plt.legend(legend, best_schedule_names, title='Schedule')
plt.savefig('Output/P3_SA_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

###### TODO: Consider the consequences of a constant temperature or changing the speed of decay or decay limit #####

'''
    Fitted Algorithm Runs

    The best performance for SA on this problem was found with max_attempts = 100, schedule = Geometric
    Taking these hyper-parameters we will run SA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
schedule = best_schedules[0]
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.simulated_annealing(p, schedule, max_attempts=75, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Simulated Annealing Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P3_SA_Best_Fitness.png', bbox_inches='tight')
plt.close()

# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Simulated Annealing on \nFlip Flop Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P3_SA_Convergence.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Simulated Annealing Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P3_SA_Wall_Time.png', bbox_inches='tight')
plt.close()
###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close()
    
# Genetic Algorithm

file = open("Output/GA_P3_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    GA has 3 main parameters that we can tune: pop_size, mutation_prob and max_attempts (which is our stopping criteria)
    We want to find the combination of these 3 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of random hill climbing for different values of max_restarts
file.write('Pop Size Analysis\n')
pop_sizes = [5, 25, 100, 200]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for s in pop_sizes:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=s, mutation_prob=0.1, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Population Size: ' + str(s) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of SA for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nPopulation Sizes on Flip Flop Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P3_GA_Pop_Size_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')


# Next we will experiment with different mutation probabilities
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for p in probs:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=200, mutation_prob=p, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Mutation Probability: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max restart value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nMutation Probabilities on Flip Flop Problem')
plt.legend(legend, probs, title='Mutation Probability')
plt.savefig('Output/P3_GA_Mutation_Probability_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect GA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=a, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of GA for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nMax Attempt Values on Flip Flop Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P3_GA_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

##### TODO Figure out a good visualization for 3 hyper-parameters #####
# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for s in pop_sizes:
    file.write('Population size: ' + str(s) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for p in probs:
        file.write('\tProbability: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
        best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem, pop_size=s, mutation_prob=p, max_attempts=25, max_iters=np.inf, curve=True, random_state=random_seeds[1])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_curves.append(len(fitness_curve))
        iter_iterations.append(fitness_curve)
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(probs))),probs)
plt.ylabel('Best Fitness')
plt.xlabel('Mutation Probability')
plt.title('Genetic Algorithm Fitness Comparison for \nHyper-Parameters on Flip Flop Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P3_GA_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for GA on this problem was found with max_attempts = 25, population = 100, mutation_probability = 0.4
    Taking these hyper-parameters we will run GA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.genetic_alg(p, pop_size=100, mutation_prob=0.4, max_attempts=100, max_iters=np.inf, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Genetic Algorithm Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P3_GA_Best_Fitness.png', bbox_inches='tight')
plt.close()
# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Genetic Algorithm on \nFlip Flop Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P3_GA_Convergence.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Genetic Algorithm Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P3_GA_Wall_Time.png', bbox_inches='tight')
plt.close()

###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close()

# MIMIC

file = open("Output/MIMIC_P3_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    GA has 3 main parameters that we can tune: pop_size, keep_pct and max_attempts (which is our stopping criteria)
    We want to find the combination of these 3 hyper-parameters that gives us a best solution to this problem
'''
First let's experiment with the performance of MIMIC for different values of population size
file.write('Pop Size Analysis\n')
pop_sizes = [5, 25, 100, 200, 500]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for s in pop_sizes:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=s, keep_pct=0.2, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Population Size: ' + str(s) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of MIMIC for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nPopulation Sizes on Flip Flop Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P3_MIMIC_Pop_Size_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')


# Next we will experiment with different keep_pct
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for p in probs:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=200, keep_pct=p, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Keep %: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max restart value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nKeep Percentages on Flip Flop Problem')
plt.legend(legend, probs, title='Keep %')
plt.savefig('Output/P3_MIMIC_Keep_Percentage_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect GA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=200, keep_pct=0.1, max_attempts=a, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of GA for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nMax Attempt Values on Flip Flop Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P3_MIMIC_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

##### TODO Figure out a good visualization for 3 hyper-parameters #####
# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for s in pop_sizes:
    file.write('Population size: ' + str(s) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for p in probs:
        best_state, best_fitness, fitness_curve = mlr.mimic(problem, pop_size=s, keep_pct=p, max_attempts=100, max_iters=np.inf, curve=True, random_state=random_seeds[0])      
        file.write('\tKeep Percentage: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_curves.append(len(fitness_curve))
        iter_iterations.append(fitness_curve)
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(probs))),probs)
plt.ylabel('Best Fitness')
plt.xlabel('Mutation Probability')
plt.title('Genetic Algorithm Fitness Comparison for \nHyper-Parameters on Flip Flop Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P3_MIMIC_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for GA on this problem was found with max_attempts = 100, population = 200, keep_pct = 0.3
    Taking these hyper-parameters we will run GA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.mimic(p, pop_size=200, keep_pct=0.5, max_attempts=100, max_iters=np.inf, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('MIMIC Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P3_MIMIC_Best_Fitness_250.png', bbox_inches='tight')
plt.close()

# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of MIMIC on \nFlip Flop Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P3_MIMIC_Convergence_250.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('MIMIC Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P3_MIMIC_Wall_Time_250.png', bbox_inches='tight')
plt.close()
###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close

# Graph Comparing Each Algorithm's Average performance on each of the 4 problem sizes
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_ff)), p1_ff, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_ff)), p2_ff, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_ff)), p3_ff, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_ff)), p4_ff, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Fitness for Each Algorithm on \nVarying Sizes of the Flip Flop Problem')
plt.tight_layout()
fig.savefig('Output/P3_ALL_Fitness.png', bbox_inches='tight')
plt.close()

# Graph comparing wall time for each of the algorithms
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_ft)), p1_ft, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_ft)), p2_ft, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_ft)), p3_ft, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_ft)), p4_ft, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Wall Clock Time for Each Algorithm on \nVarying Sizes of the Flip Flop Problem')
plt.tight_layout()
fig.savefig('Output/P3_ALL_Wall_Clock.png', bbox_inches='tight')
plt.close()

# Graph comparing iterations to converge for each of the algorithms
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_fi)), p1_fi, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_fi)), p2_fi, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_fi)), p3_fi, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_fi)), p4_fi, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Iterations to Convergence for Each Algorithm on \nVarying Sizes of the Flip Flop Problem')
plt.tight_layout()
fig.savefig('Output/P3_ALL_Iterations.png', bbox_inches='tight')
plt.close() """

'''
KNAPSACK
'''
N = [50, 100, 125, 150]
weights = [np.random.randint(1,25,size=n) for n in N]
values = [np.random.randint(1,25,size=n) for n in N]
n_train = 25
weight_train = np.random.randint(1,25, size=n_train)
value_train = np.random.randint(1,25, size=n_train)
fitness_train = mlr.Knapsack(weight_train, value_train)
problem_train = mlr.DiscreteOpt(length=n_train, fitness_fn=fitness_train, maximize=True, max_val=2)
problems = []
problem_train.set_mimic_fast_mode(True)
for i in range(len(N)):
    fitness = mlr.Knapsack(weights[i], values[i])
    problem = mlr.DiscreteOpt(length=N[i], fitness_fn=fitness, maximize=True, max_val=2)
    problems.append(problem)
    
# problems and data structures for fitted algorithm comparison
p1_fi, p2_fi, p3_fi, p4_fi = [],[],[],[]
p1_ff, p2_ff, p3_ff, p4_ff = [],[],[],[]
p1_ft, p2_ft, p3_ft, p4_ft = [],[],[],[]
final_curves = []
final_fitnesses = []
final_times = []
problem_names = ['Knapsack N=50', 'Knapsack N=100', 'Knapsack N=125', 'Knapsack N=150']
# Random Hill Climbing


'''
    HYPER-PARAMETER TUNING
    
    RHC has 2 main parameters that we can tune: max_restarts and max_attempts (which is our stopping criteria)
    We want to find the combination of these 2 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of random hill climbing for different values of max_restarts
max_restarts = [1, 5, 10, 15, 20, 25]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for r in max_restarts:
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem_train, max_attempts=n_train, max_iters=np.inf, restarts=r, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of SA for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Random Hill Climbing Fitness of Various \nRestart Values on Knapsack Problem')
plt.legend(legend, max_restarts, title='Max Restarts')
plt.savefig('Output/P4_RHC_Max_Restarts_fitness.png')
plt.close()

# Now let's experiment with just the max_attempts parameter to determine how different values effect RHC on this problem
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem_train, max_attempts=a, max_iters=np.inf, restarts=1, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Random Hill Climbing Fitness of Various \nRestart Values on Knapsack Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P4_RHC_Max_Attempts_fitness.png')
plt.close()


# Finally let's see how different combinations of these hyperparameters work together
max_attempts = [5, 25, 50, 75, 100]
max_restarts = [1, 10, 15, 50]
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for r in max_restarts:
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for a in max_attempts:
        best_state, best_fitness, fitness_curve = mlr.random_hill_climb(problem_train, max_attempts=a, max_iters=np.inf, restarts=r, init_state=None, curve=True, random_state=random_seeds[0])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_iterations.append(len(fitness_curve))
        iter_curves.append(fitness_curve)
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(max_attempts))),max_attempts)
plt.ylabel('Best Fitness')
plt.xlabel('Max Attempts')
plt.title('Random Hill Climbing Fitness Comparison for \nHyper-Parameters on Knapsack Problem')
plt.legend(legend, max_restarts, title='Max Restarts')
plt.savefig('Output/P4_RHC_hyper_parameter_combination.png')
plt.close()

'''
    Fitted Algorithm Runs

    The best performance for RHC on this problem was found with max_attempts = 25, max_restarts = 50
    Taking these hyper-parameters we will run RHC on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.random_hill_climb(p, max_attempts=100, max_iters=np.inf, restarts=50, init_state=None, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Random Hill Climbing Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P4_RHC_Best_Fitness_1000.png', bbox_inches='tight')
plt.close()

# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Random Hill Climbing on \nKnapsack Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P4_RHC_Convergence_1000.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Random Hill Climbing Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P4_RHC_Wall_Time_1000.png', bbox_inches='tight')
plt.close()
###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

    
# Simulated Annealing

# Open a txt file to log data in
file = open("Output/SA_P3_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    SA has 2 main parameters that we can tune: schedule (which is our decay schedule) and max_attempts (which is our stopping criteria)
    We want to find the combination of these 2 hyper-parameters that gives us a best solution to this problem
'''
# First let's explore the performance of different decay schedules on the default SA algorithm from MLRose
schedules_geom = [mlr.GeomDecay(), mlr.GeomDecay(min_temp=0.5), mlr.GeomDecay(decay=0.99999), mlr.GeomDecay(decay=0.99999,min_temp=0.5)]
schedule_names_geom = ['Geom1', 'Geom2', 'Geom3', 'Geom4']
fitness_curves_geom = [] # store fitness curve for each decay function
best_fitnesses_geom = [] # Store best fitness for each function

schedules_arith = [mlr.ArithDecay(), mlr.ArithDecay(min_temp=0.5), mlr.ArithDecay(decay=0.0000001), mlr.ArithDecay(decay=0.0000001,min_temp=0.5)]
schedule_names_arith = ['Arith1', 'Arith2', 'Arith3', 'Arith4']
fitness_curves_arith = [] # store fitness curve for each decay function
best_fitnesses_arith = [] # Store best fitness for each function

schedules_exp = [mlr.ExpDecay(), mlr.ExpDecay(min_temp=0.5), mlr.ExpDecay(exp_const=0.00005), mlr.ExpDecay(exp_const=0.00005,min_temp=0.5)]
schedule_names_exp = ['Exp1', 'Exp2', 'Exp3', 'Exp4']
fitness_curves_exp = [] # store fitness curve for each decay function
best_fitnesses_exp = [] # Store best fitness for each function

for i in range(len(schedules_geom)):
    schedule = schedules_geom[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem_train, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_geom[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_geom.append(fitness_curve)
    best_fitnesses_geom.append(best_fitness)
for i in range(len(schedules_arith)):
    schedule = schedules_arith[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem_train, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_arith[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_arith.append(fitness_curve)
    best_fitnesses_arith.append(best_fitness)
for i in range(len(schedules_exp)):
    schedule = schedules_exp[i]
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem_train, schedule, max_attempts=10, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    # log the best fitness for each schedule
    file.write('Schedule: ' + schedule_names_exp[i] + ' with best fitness value: ' + str(best_fitness) + '\n')
    fitness_curves_exp.append(fitness_curve)
    best_fitnesses_exp.append(best_fitness)
# Plot curve showing performance of SA for each decay function
fig, axs = plt.subplots(2,2, figsize=(15,15))
fig.suptitle('Simulated Annealing Fitness of Various \nDecay Functions on Knapsack Problem')
legend = []
for y_vals in fitness_curves_geom:
    ln, = axs[0,0].plot(y_vals)
    legend.append(ln)
axs[0,0].set_title('Geometric Decay')
axs[0,0].legend(legend, schedule_names_geom, title='Decay Function')
legend = []
for y_vals in fitness_curves_arith:
    ln, = axs[0,1].plot(y_vals)
    legend.append(ln)
axs[0,1].set_title('Arithmetic Decay')
axs[0,1].legend(legend, schedule_names_arith, title='Decay Function')
legend = []
for y_vals in fitness_curves_exp:
    ln, = axs[1,0].plot(y_vals)
    legend.append(ln)
axs[1,0].set_title('Exponential Decay')
axs[1,0].legend(legend, schedule_names_exp, title='Decay Function')
axs[1,1].axis('off')
for ax in axs.flat:
    ax.set(xlabel='Iteration', ylabel='Fitness')
for ax in axs.flat:
    ax.label_outer()
plt.savefig('Output/P4_SA_Decay_Function_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect SA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, max_attempts=a, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Simulated Annealing Fitness of Various \nRestart Values on Knapsack Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P4_SA_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Finally let's see how different combinations of these hyperparameters work together
best_schedules = [mlr.GeomDecay(), mlr.GeomDecay(min_temp=0.5), mlr.ExpDecay(min_temp=0.5)]
best_schedule_names = ['Geo1', 'Geo2', 'Exp2']
file.write('Hyper Parameter Combinations\n')
max_attempts = [5, 25, 50, 75, 100]
max_restarts = [1, 10, 15, 50]
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for i in range(len(best_schedules)):
    schedule = best_schedules[i]
    file.write('Schedule: ' + best_schedule_names[i] + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for a in max_attempts:
        best_state, best_fitness, fitness_curve = mlr.simulated_annealing(problem, schedule, max_attempts=a, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[1])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_iterations.append(len(fitness_curve))
        iter_curves.append(fitness_curve)
        file.write('\tMax Attempts: ' + str(r) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(max_attempts))),max_attempts)
plt.ylabel('Best Fitness')
plt.xlabel('Max Attempts')
plt.title('Simulated Annealing Fitness Comparison for \nHyper-Parameters on Knapsack Problem')
plt.legend(legend, best_schedule_names, title='Schedule')
plt.savefig('Output/P4_SA_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

###### TODO: Consider the consequences of a constant temperature or changing the speed of decay or decay limit #####

'''
    Fitted Algorithm Runs

    The best performance for SA on this problem was found with max_attempts = 100, schedule = Geometric
    Taking these hyper-parameters we will run SA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
schedule = best_schedules[2]
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.simulated_annealing(p, schedule, max_attempts=100, max_iters=np.inf, init_state=None, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Simulated Annealing Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P4_SA_Best_Fitness.png', bbox_inches='tight')
plt.close()

# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Simulated Annealing on \nKnapsack Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P4_SA_Convergence.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Simulated Annealing Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P4_SA_Wall_Time.png', bbox_inches='tight')
plt.close()
###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close()
    
# Genetic Algorithm

file = open("Output/GA_P3_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    GA has 3 main parameters that we can tune: pop_size, mutation_prob and max_attempts (which is our stopping criteria)
    We want to find the combination of these 3 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of random hill climbing for different values of max_restarts
file.write('Pop Size Analysis\n')
pop_sizes = [5, 25, 100, 200]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for s in pop_sizes:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem_train, pop_size=s, mutation_prob=0.1, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Population Size: ' + str(s) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of SA for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nPopulation Sizes on Knapsack Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P4_GA_Pop_Size_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')


# Next we will experiment with different mutation probabilities
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for p in probs:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem_train, pop_size=200, mutation_prob=p, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Mutation Probability: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max restart value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nMutation Probabilities on Knapsack Problem')
plt.legend(legend, probs, title='Mutation Probability')
plt.savefig('Output/P4_GA_Mutation_Probability_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect GA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem_train, pop_size=200, mutation_prob=0.1, max_attempts=a, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of GA for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('Genetic Algorithm Fitness of Various \nMax Attempt Values on Knapsack Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P4_GA_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

##### TODO Figure out a good visualization for 3 hyper-parameters #####
# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for s in pop_sizes:
    file.write('Population size: ' + str(s) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for p in probs:
        file.write('\tProbability: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
        best_state, best_fitness, fitness_curve = mlr.genetic_alg(problem_train, pop_size=s, mutation_prob=p, max_attempts=25, max_iters=np.inf, curve=True, random_state=random_seeds[1])
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_curves.append(len(fitness_curve))
        iter_iterations.append(fitness_curve)
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(probs))),probs)
plt.ylabel('Best Fitness')
plt.xlabel('Mutation Probability')
plt.title('Genetic Algorithm Fitness Comparison for \nHyper-Parameters on Knapsack Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P4_GA_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for GA on this problem was found with max_attempts = 25, population = 200, mutation_probability = 0.75
    Taking these hyper-parameters we will run GA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.genetic_alg(p, pop_size=200, mutation_prob=0.1, max_attempts=25, max_iters=np.inf, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Genetic Algorithm Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P4_GA_Best_Fitness.png', bbox_inches='tight')
plt.close()
# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of Genetic Algorithm on \nKnapsack Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P4_GA_Convergence.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Genetic Algorithm Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P4_GA_Wall_Time.png', bbox_inches='tight')
plt.close()

###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close()

# MIMIC

file = open("Output/MIMIC_P4_log.txt","w")

'''
    HYPER-PARAMETER TUNING
    
    GA has 3 main parameters that we can tune: pop_size, keep_pct and max_attempts (which is our stopping criteria)
    We want to find the combination of these 3 hyper-parameters that gives us a best solution to this problem
'''
# First let's experiment with the performance of MIMIC for different values of population size
file.write('Pop Size Analysis\n')
pop_sizes = [5, 25, 100, 200, 500]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for s in pop_sizes:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem_train, pop_size=s, keep_pct=0.2, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Population Size: ' + str(s) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of MIMIC for each decay function
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nPopulation Sizes on Knapsack Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P4_MIMIC_Pop_Size_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')


# Next we will experiment with different keep_pct
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for p in probs:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem_train, pop_size=200, keep_pct=p, max_attempts=10, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Keep %: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of RHC for each max restart value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nKeep Percentages on Knapsack Problem')
plt.legend(legend, probs, title='Keep %')
plt.savefig('Output/P4_MIMIC_Keep_Percentage_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

# Now let's experiment with just the max_attempts parameter to determine how different values effect GA on this problem
file.write('Max Attempts Analysis\n')
max_attempts = [5, 25, 50, 75, 100]
fitness_curves = [] # store fitness curve for each decay function
best_fitnesses = [] # Store best fitness for each function
for a in max_attempts:
    best_state, best_fitness, fitness_curve = mlr.mimic(problem_train, pop_size=200, keep_pct=0.1, max_attempts=a, max_iters=np.inf, curve=True, random_state=random_seeds[0])
    fitness_curves.append(fitness_curve)
    # log the best fitness for each max_restart value
    file.write('Max Attempts: ' + str(a) + ' with best fitness value: ' + str(best_fitness) + '\n')
    best_fitnesses.append(best_fitness)
# Plot curve showing performance of GA for each max attempts value
legend = []
for y_vals in fitness_curves:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.ylabel('Fitness')
plt.xlabel('Iteration')
plt.title('MIMIC Fitness of Various \nMax Attempt Values on Knapsack Problem')
plt.legend(legend, max_attempts, title='Max Attempts')
plt.savefig('Output/P4_MIMIC_Max_Attempts_fitness.png')
plt.close()
file.write('------------------------------------------------------------------\n')

##### TODO Figure out a good visualization for 3 hyper-parameters #####
# Finally let's see how different combinations of these hyperparameters work together
file.write('Hyper Parameter Combinations\n')
best_states = []
best_fitnesses = []
fitness_curves = []
iterations = []
for s in pop_sizes:
    file.write('Population size: ' + str(s) + '\n')
    iter_states = []
    iter_fitnesses = []
    iter_curves = []
    iter_iterations = []
    for p in probs:
        best_state, best_fitness, fitness_curve = mlr.mimic(problem_train, pop_size=s, keep_pct=p, max_attempts=100, max_iters=np.inf, curve=True, random_state=random_seeds[0])      
        file.write('\tKeep Percentage: ' + str(p) + ' with best fitness value: ' + str(best_fitness) + '\n')
        iter_states.append(best_state)
        iter_fitnesses.append(best_fitness)
        iter_curves.append(len(fitness_curve))
        iter_iterations.append(fitness_curve)
    best_states.append(iter_states)
    best_fitnesses.append(iter_fitnesses)
    fitness_curves.append(iter_curves)
    iterations.append(iter_iterations)
# Plot curve for max_attempts and max_restarts
legend = []
for y_vals in best_fitnesses:
    ln, = plt.plot(y_vals)
    legend.append(ln)
plt.xticks(list(range(len(probs))),probs)
plt.ylabel('Best Fitness')
plt.xlabel('Mutation Probability')
plt.title('Genetic Algorithm Fitness Comparison for \nHyper-Parameters on Knapsack Problem')
plt.legend(legend, pop_sizes, title='Population Size')
plt.savefig('Output/P4_MIMIC_hyper_parameter_combination.png')
plt.close()
file.write('------------------------------------------------------------------\n')

'''
    Fitted Algorithm Runs

    The best performance for GA on this problem was found with max_attempts = 100, population = 200, keep_pct = 0.3
    Taking these hyper-parameters we will run GA on different sizes of the problem 10 times and show the average performance for each problem size as well as the noise
'''
all_fitness_curves = [] # store fitness curve for each decay function
all_best_fitnesses = [] # Store best fitness for each 
all_times = []
all_iterations = []
for i in range(len(problems)):
    fitness_curves = []
    best_fitnesses = []
    running_time = []
    iterations = []
    file.write(problem_names[i] + ': \n')
    p = problems[i]
    p.set_mimic_fast_mode(True)
    for j in range(10):
        start_time = time.time()
        best_state, best_fitness, fitness_curve = mlr.mimic(p, pop_size=200, keep_pct=0.1, max_attempts=100, max_iters=np.inf, curve=True, random_state=random_seeds[j+1])
        end_time = time.time()
        running_time.append(end_time-start_time)
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(best_fitness)
        iterations.append(len(fitness_curve))
    all_fitness_curves.append(fitness_curves)
    all_best_fitnesses.append(best_fitnesses)
    all_times.append(running_time)
    all_iterations.append(iterations)
# Add average best_fitness to all_best_fitnesses
for bf in all_best_fitnesses:
    bf.append(np.mean(bf))
# Plot the results of all 3 problems in a 2x2 subplot of bar charts with summary in bottom right
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
axs[0,0].barh(np.arange(len(all_best_fitnesses[0])), all_best_fitnesses[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_best_fitnesses[1])), all_best_fitnesses[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_best_fitnesses[2])), all_best_fitnesses[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_best_fitnesses[3])), all_best_fitnesses[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('MIMIC Best Fitness over 10 Trials\n(Average Performance in Blue)')
plt.tight_layout()
fig.savefig('Output/P4_MIMIC_Best_Fitness.png', bbox_inches='tight')
plt.close()

# Plot the fitness curves for each trial of each problem as a subplot
fig, axs = plt.subplots(2,2)
fig.suptitle('Convergence of MIMIC on \nKnapsack Problems of Varying Size')
for y_vals in all_fitness_curves[0]:
    axs[0,0].plot(y_vals)
axs[0,0].set_title(problem_names[0])
for y_vals in all_fitness_curves[1]:
    axs[0,1].plot(y_vals)
axs[0,1].set_title(problem_names[1])
for y_vals in all_fitness_curves[2]:
    axs[1,0].plot(y_vals)
axs[1,0].set_title(problem_names[2])
for y_vals in all_fitness_curves[3]:
    axs[1,1].plot(y_vals)
axs[1,1].set_title(problem_names[3])
fig.tight_layout()
fig.savefig('Output/P4_MIMIC_Convergence.png', bbox_inches='tight')
plt.close()
##### TODO: Consider setting constant x-axis values across all 3 subplots ######

# Add average best_fitness to all_best_fitnesses
for t in all_times:
    t.append(np.mean(t))
# Plot wall time for each problem size
max_time = 0
for times in all_times:
    if max(times) > max_time:
        max_time = max(times)
colors = 10*['gray']
colors.append('blue')
fig, axs = plt.subplots(2,2)
plt.setp(axs, xlim=(0,max_time))
axs[0,0].barh(np.arange(len(all_times[0])), all_times[0], color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(all_times[1])), all_times[1], color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(all_times[2])), all_times[2], color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(all_times[3])), all_times[3], color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('MIMIC Wall Time over 10 Trials\n(Average Time in Blue)')
plt.tight_layout()
fig.savefig('Output/P4_MIMIC_Wall_Time.png', bbox_inches='tight')
plt.close()
###### TODO Decide what you want to do with final fitness curves #####
# Add to final lists
p1_fi.append(np.mean(all_iterations[0]))
p2_fi.append(np.mean(all_iterations[1]))
p3_fi.append(np.mean(all_iterations[2]))
p4_fi.append(np.mean(all_iterations[3]))
p1_ff.append(all_best_fitnesses[0][-1])
p2_ff.append(all_best_fitnesses[1][-1])
p3_ff.append(all_best_fitnesses[2][-1])
p4_ff.append(all_best_fitnesses[3][-1])
p1_ft.append(all_times[0][-1])
p2_ft.append(all_times[1][-1])
p3_ft.append(all_times[2][-1])
p4_ft.append(all_times[3][-1])

file.close

# Graph Comparing Each Algorithm's Average performance on each of the 4 problem sizes
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_ff)), p1_ff, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_ff)), p2_ff, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_ff)), p3_ff, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_ff)), p4_ff, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Fitness for Each Algorithm on \nVarying Sizes of the Knapsack Problem')
plt.tight_layout()
fig.savefig('Output/P4_ALL_Fitness.png', bbox_inches='tight')
plt.close()

# Graph comparing wall time for each of the algorithms
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_ft)), p1_ft, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_ft)), p2_ft, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_ft)), p3_ft, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_ft)), p4_ft, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Wall Clock Time for Each Algorithm on \nVarying Sizes of the Knapsack Problem')
plt.tight_layout()
fig.savefig('Output/P4_ALL_Wall_Clock.png', bbox_inches='tight')
plt.close()

# Graph comparing iterations to converge for each of the algorithms
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(2,2)
plt.setp(axs, yticks=[0,1,2,3])
plt.setp(axs, yticklabels=['RHC', 'SA', 'GA', 'MIMIC'])
axs[0,0].barh(np.arange(len(p1_fi)), p1_fi, color=colors)
axs[0,0].set_title(problem_names[0])
axs[0,1].barh(np.arange(len(p2_fi)), p2_fi, color=colors)
axs[0,1].set_title(problem_names[1])
axs[1,0].barh(np.arange(len(p3_fi)), p3_fi, color=colors)
axs[1,0].set_title(problem_names[2])
axs[1,1].barh(np.arange(len(p4_fi)), p4_fi, color=colors)
axs[1,1].set_title(problem_names[3])
fig.suptitle('Average Iterations to Convergence for Each Algorithm on \nVarying Sizes of the Knapsack Problem')
plt.tight_layout()
fig.savefig('Output/P4_ALL_Iterations.png', bbox_inches='tight')
plt.close()