from evolution import *
import gym
#from gym.spaces import *
#from random import *
#import tensorflow as tf
#from nets import *
#from learner import *

def h(x):
    return 0 if x<0 else 1

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env, './evolution/cartpole-v0-2', force=True)
    policy_f = lambda w, obs: int(h(np.dot(w[0:4], obs) + w[4]))
    (te, _, _) = evolve_env(env, 
                            policy_f,
                            np.asarray([0,0,0,0,0]), 
                            lambda w, spawn, s: gaussian_perturb(w,spawn, stddev=1/np.sqrt(s)), 
                            normalized_avg, 
                            alpha=1,
                            spawn=50,
                            stages=10,
                            print_every=1, 
                            max_steps=1000,
                            eval_trials=100)
    #print("Running:")
    #for i in range(1000):
    #    print("Reward %f" % env_f(env, policy_f, te, max_steps=1000))
    env.close()
    gym.upload('./evolution/cartpole-v0-2', api_key='API_KEY')

