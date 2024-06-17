import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
import time
from UAV_env import UAVEnv
# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Define constants
MAX_EPISODES = 1000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 5000
BATCH_SIZE = 64

# Initialize environment and dimensions
env = UAVEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

def tune_hyperparameter(values, fixed_params, param_name):
    best_reward = -np.inf
    best_value = None
    results = []

    for value in values:
        params = fixed_params.copy()
        params[param_name] = value
        avg_reward = train_ddpg(params['lr_a'], params['lr_c'], params['gamma'], params['tau'])
        results.append((params['lr_a'], params['lr_c'], params['gamma'], params['tau'], avg_reward))
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_value = value
        
        # Save results to a text file
        with open('ddpg_results.txt', 'a') as file:
            file.write(f"LR_A: {params['lr_a']}, LR_C: {params['lr_c']}, GAMMA: {params['gamma']}, TAU: {params['tau']}, AVG_REWARD: {avg_reward}\n")
    return best_value, best_reward, results

def train_ddpg(lr_a, lr_c, gamma, tau):
    ddpg = DDPG(a_dim, s_dim, a_bound, lr_a, lr_c, gamma, tau)
    ep_reward_list = []

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            a = ddpg.choose_action(s)
            s_, r, is_terminal, _, _, _ = env.step(a)
            ddpg.store_transition(s, a, r, s_)
            if ddpg.pointer > MEMORY_CAPACITY:
                ddpg.learn()
            s = s_
            ep_reward += r
            if is_terminal:
                break
        ep_reward_list.append(ep_reward)

    avg_reward = np.mean(ep_reward_list)
    return avg_reward

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, lr_a, lr_c, gamma, tau):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.compat.v1.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')

        with tf.compat.v1.variable_scope('Actor', reuse=tf.compat.v1.AUTO_REUSE):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope('Critic', reuse=tf.compat.v1.AUTO_REUSE):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_replace = [tf.compat.v1.assign(t, (1 - tau) * t + tau * e)
                     for t, e in zip(self.at_params, self.ae_params)] + \
                    [tf.compat.v1.assign(t, (1 - tau) * t + tau * e)
                     for t, e in zip(self.ct_params, self.ce_params)]


        q_target = self.R + gamma * q_
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)
        self.atrain = tf.compat.v1.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        self.sess.run(self.soft_replace)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.compat.v1.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            a = tf.compat.v1.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            n_l1 = 400
            w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.compat.v1.layers.dense(net, 1, trainable=trainable)


# Define the ranges for hyperparameters
lr_a_values = [0.001, 0.01, 0.1]
lr_c_values = [0.002, 0.02, 0.2]
gamma_values = [0.9, 0.99, 0.999]
tau_values = [0.01, 0.005, 0.001]

# Initial fixed parameters
fixed_params = {
    'lr_a': 0.001,
    'lr_c': 0.002,
    'gamma': 0.9,
    'tau': 0.01
}

# Tune learning rate for actor
best_lr_a, _, lr_a_results = tune_hyperparameter(lr_a_values, fixed_params, 'lr_a')
fixed_params['lr_a'] = best_lr_a

# Tune learning rate for critic
best_lr_c, _, lr_c_results = tune_hyperparameter(lr_c_values, fixed_params, 'lr_c')
fixed_params['lr_c'] = best_lr_c

# Tune gamma
best_gamma, _, gamma_results = tune_hyperparameter(gamma_values, fixed_params, 'gamma')
fixed_params['gamma'] = best_gamma

# Tune tau
best_tau, _, tau_results = tune_hyperparameter(tau_values, fixed_params, 'tau')
fixed_params['tau'] = best_tau

print("Best Hyperparameters: ", fixed_params)

# Plotting the results
results = np.array(lr_a_results + lr_c_results + gamma_results + tau_results)
avg_rewards = results[:, 4]
plt.figure(figsize=(12, 8))
plt.plot(avg_rewards, label='Average Reward')
plt.xlabel('Hyperparameter Set')
plt.ylabel('Average Reward')
plt.title('Hyperparameter Tuning Results')
plt.legend()
plt.grid(True)
plt.show()
