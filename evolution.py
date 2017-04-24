#from gym.spaces import *
import numpy as np
from utils import *
#from random import *
#import tensorflow as tf
#from nets import *
#from learner import *

#Perturbation functions
def gaussian_perturb(te,spawn=100,stddev=1):
    sh = list(te.shape)
    if isinstance(stddev, list):
        #return [te + np.random.randn(scale=stddev[i], size=sh) for i in range(spawn)]
        perts = np.random.normal(scale=stddev, size=sh)
    else:
        perts = [np.random.normal(scale=stddev, size=sh) for i in range(spawn)] 
    return (perts, te + perts)

#Combine functions
#give each the weight of rewards
def simple_avg(params, rewards):
    return np.dot(rewards, params)/sum(rewards)

#normalize rewards to be gaussian first
def normalized_avg(params, rewards):
    std = np.std(rewards)
    norm_rewards = (rewards - np.mean(rewards)) / (std if std>0 else 1)
    return np.dot(norm_rewards, params)/len(params)

#no memory, rely only on observation  
def env_f(env, policy_f, te, max_steps=1000, render=False):
    observation = env.reset()
    tot_r = 0
    for t in range(max_steps):
        if render:
            env.render()
        action = policy_f(te, observation)
        observation, reward, done, info = env.step(action)
        tot_r += reward
        if done:
            break
    return tot_r

def evolve_env(env, policy_f, te, perturb_f, combine_f, alpha=1, trials=20, spawn=100, stages=100, verbosity=1, render=False, max_steps=1000, print_every=20, eval_trials=1):
    return evolve(lambda te1: env_f(env, policy_f, te1, max_steps, render), te, perturb_f, combine_f, alpha, trials, spawn, stages, verbosity, print_every, eval_trials)

def evolve(f, te, perturb_f, combine_f, alpha=1, trials=1, spawn=100, stages=100, verbosity=1, print_every=20, eval_trials=1):
    tes = []
    rews = []
    for s in range(1,stages+1):
        (diffs, children) = perturb_f(te, spawn, s)
        printv(children, verbosity, 2)
        rewards = [np.mean([f(c) for i in range(trials)]) for c in children] # do i times
        step = combine_f(diffs, rewards)
        #print(step)
        #print(alpha*step)
        te = te + alpha*step
        rew = np.mean([f(te) for i in range(eval_trials)])
        if (s%print_every==0):
            printv("Stage %d, Parameters %s, Reward %f" % (s, str(te), rew), verbosity, 1)
        tes += [te]
        rews += [rew]
    return (te, tes, rews)

if __name__=='__main__':
    solution = np.array([0.5, 0.1, -0.3])
    evolve(lambda w: -np.sum(np.square(solution - w)), 
           #np.random.randn(3),
           np.asarray([1.7, .4, 1]),
           lambda w, spawn, s: gaussian_perturb(w, spawn,0.1),
           normalized_avg,
           alpha=0.1, 
           spawn=50,
           stages=300,
           print_every=20)
           
