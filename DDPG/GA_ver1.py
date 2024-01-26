"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.

Using:
tensorflow 1.14.0
gym 0.15.3
"""


import numpy as np
import math
import copy
import matplotlib.pyplot as plt
#-----------------------------------------------------------#
import tensorflow as tf
import numpy as np
from UAV_env import UAVEnv
import time
import matplotlib.pyplot as plt
from state_normalization import StateNormalization
tf.compat.v1.disable_eager_execution()

#####################  hyper parameters  ####################

MAX_EPISODES = 1000
# MAX_EPISODES = 50000
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
# LR_A = 0.1  # learning rate for actor
# LR_C = 0.2  # learning rate for critic
GAMMA = 0.001  # optimal reward discount
# GAMMA = 0.999  # reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01
# MEMORY_CAPACITY = 5000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
OUTPUT_GRAPH = False

#-----------------------------------------------------------#

chromo = 50
nmbpopu = 64
populationLimit = 128
mutation_rate = 0.3
mutation_rate_1 = 0.3
mutation_rate_2 = 0.3
generations = 1000
x = [] 
e = 1e-3
population = []
bestState = None


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(ddpg, eval_episodes=10):
    # eval_env = gym.make(env_name)
    eval_env = UAVEnv()
    # eval_env.seed(seed + 100)
    avg_reward = 0.
    for i in range(eval_episodes):
        state = eval_env.reset()
        # while not done:
        for j in range(int(len(eval_env.UE_loc_list))):
            action = ddpg.choose_action(state)
            action = np.clip(action, *a_bound)
            state, reward = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # memory里存放当前和下一个state，动作和奖励 - Trạng thái, hành động và phần thưởng hiện tại và tiếp theo được lưu trữ trong bộ nhớ
        self.pointer = 0
        self.sess = tf.compat.v1.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')

        with tf.compat.v1.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q) #hàm mất mát
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]

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
        # transition = np.hstack((s, [a], [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = tf.compat.v1.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.compat.v1.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.compat.v1.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.compat.v1.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.compat.v1.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.compat.v1.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.compat.v1.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.compat.v1.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################


class Indv(object):
    def __init__(self):
        self.genes = []
        self._ep_reward = 0
        self._ep_reward_list = []
    def _add_state(self):
        env = UAVEnv()
        #self.genes = np.append(self.genes,copy.deepcopy(env))
        for i in range(chromo):
            action = np.random.uniform(-1,1,size = (env.M,))
            env.act = (action + 1) / 2
            _env = copy.deepcopy(env)
            s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(action)
            _env.sum_task_size = env.sum_task_size
            _env.e_battery_uav = env.e_battery_uav
            self._ep_reward += r
            self._ep_reward_list = np.append(self._ep_reward_list,r)
            self.genes = np.append(self.genes,_env)
            if is_terminal or step_redo:
                break

def crossOver(a,b):
    x = len(a.genes) if len(a.genes) < len(b.genes) else len(b.genes)
    i1 = np.random.randint(x)
    i2 = (i1 + 1 + np.random.randint(x - 1)) % x
    n1 = copy.deepcopy(a)
    n2 = copy.deepcopy(b)
    if i1 < i2:
        i = i1
        j = i2
    else:
        i = i2
        j = i1
    while i != j:
        n1.genes[i] = b.genes[i]
        n2.genes[i] = a.genes[i]
        i = i + 1
    """...
    sum1 = 0
    sum2 = 0
    for k in range(len(a)):
        task_size1 = n1.genes[k].task_list[n1.genes[k].action[0] * n1.genes[k].M] 
        task_size2 = n2.genes[k].task_list[n2.genes[k].action[0] * n2.genes[k].M] 
        sum1 += task_size1
        sum2 += task_size2
    for k in range(len(a)):
        n1.genes[k].task_list[n1.genes[k].action[0] * n1.genes[k].M] *= n1.genes[0].sum_task_size/sum1
        n2.genes[k].task_list[n2.genes[k].action[0] * n2.genes[k].M] *= n2.genes[0].sum_task_size/sum2
    ..."""
    update(n1)
    update(n2)
    return n1,n2



def update(self):
    env_ = UAVEnv()
    self._ep_reward = 0
    self._ep_reward_list = []
    # self_ = []
    for i in range(len(self.genes) - 1):
        env_ = copy.deepcopy(self.genes[i])
        if i == 0:
            env_.e_battery_uav = 50000
            env_.sum_task_size = 100 * 1048576
        else:
            env_.e_battery_uav = self.genes[i-1].e_battery_uav
            env_.sum_task_size = self.genes[i-1].sum_task_size

        loc_uav_after_fly_x = self.genes[i+1].loc_uav[0]
        loc_uav_after_fly_y = self.genes[i+1].loc_uav[1]

        dx_uav = loc_uav_after_fly_x - env_.loc_uav[0]
        dy_uav = loc_uav_after_fly_y - env_.loc_uav[1]

        dis_fly = np.sqrt(dx_uav * dx_uav + dy_uav * dy_uav)
        dis_fly = dis_fly + e
        theta = math.acos(dx_uav / dis_fly)
        # new_act = np.array(env_.act)
        # new_act[0][2] = dis_fly / (env_.flight_speed * env_.t_fly)  
        # new_act[0][1] = theta / (np.pi * 2)
        env_.act[2] = dis_fly / (env_.flight_speed * env_.t_fly)
        env_.act[1] = theta / (np.pi * 2)
        # env_.act = new_act
        e_fly = (dis_fly / env_.t_fly) ** 2 * env_.m_uav * env_.t_fly * 0.5

        if env_.act[0] == 1:
            ue_id = env_.M - 1
        else:
            ue_id = int(env_.M * env_.act[0])
        offloading_ratio = env_.act[3]  # ue卸载率 - tỷ lệ giảm tải
        task_size = env_.task_list[ue_id]
        block_flag = env_.block_flag_list[ue_id]
        t_server = offloading_ratio * task_size / (env_.f_uav / env_.s)  
        e_server = env_.r * env_.f_uav ** 3 * t_server


        if env_.sum_task_size == 0:  # 计算任务全部完成
            is_terminal = True
            reward = 0
            #self._ep_reward += 0
        elif env_.sum_task_size - task_size < 0:  # 最后一步计算任务和ue的计算任务不匹配 - Nhiệm vụ tính toán của bước cuối cùng không khớp với nhiệm vụ tính toán của ue
            #is_terminal = True
            self.task_list = np.ones(env_.M) * env_.sum_task_size
            reward = 0
            #self._ep_reward += 0
        elif loc_uav_after_fly_x < 0 or loc_uav_after_fly_x > env_.ground_width or loc_uav_after_fly_y < 0 or loc_uav_after_fly_y > env_.ground_length:  # uav位置不对 - vị trí uav sai
            # 如果超出边界，则飞行距离dist置零
            delay = env_.com_delay(env_.loc_ue_list[ue_id], env_.loc_uav, offloading_ratio, task_size, block_flag)  # 计算delay
            reward = -delay
            #self._ep_reward += -delay
            # 更新下一时刻状态 - cập nhật trạng thái tiếp theo
            env_.e_battery_uav = env_.e_battery_uav - e_server  # uav 剩余电量 - uav năng lượng còn lại
            env_.sum_task_size -= env_.task_list[ue_id]
            # self.reset2(delay, self.loc_uav[0], self.loc_uav[1], offloading_ratio, task_size, ue_id)
        elif env_.e_battery_uav < e_fly or env_.e_battery_uav - e_fly < e_server:  # uav电量不能支持计算 - Nguồn UAV không thể hỗ trợ tính toán
            delay = env_.com_delay(env_.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                   0, task_size, block_flag)  # 计算delay ~ Tính toán độ trễ
            reward = -delay
            env_.sum_task_size -= env_.task_list[ue_id]
            #self._ep_reward += -delay
            # self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, 0, task_size, ue_id)
        else:  # 电量支持飞行,且计算任务合理,且计算任务能在剩余电量内计算 ~ Dung lượng pin hỗ trợ chuyến bay, các tác vụ tính toán hợp lý và các tác vụ tính toán có thể được tính toán trong dung lượng pin còn lại.
            delay = env_.com_delay(env_.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                   offloading_ratio, task_size, block_flag)  # 计算delay
            reward = -delay
            #self._ep_reward += -delay
            # 更新下一时刻状态
            env_.e_battery_uav = env_.e_battery_uav - e_fly - e_server  # uav 剩余电量
            #env_.loc_uav[0] = loc_uav_after_fly_x  # uav 飞行后的位置
            #env_.loc_uav[1] = loc_uav_after_fly_y
            env_.sum_task_size -= env_.task_list[ue_id]
            # self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, offloading_ratio, task_size,ue_id)   # 重置ue任务大小，剩余总任务大小，ue位置，并记录到文件
        #print(reward)
        self._ep_reward = self._ep_reward + reward
        #print(reward)
        self._ep_reward_list = np.append(self._ep_reward_list,reward)
        self.genes[i] = copy.deepcopy(env_)
        #return env_._get_obs(env_.act), reward, is_terminal, step_redo, offloading_ratio_change, reset_dist
    reward = 0
    self._ep_reward = self._ep_reward + reward
    self._ep_reward_list = np.append(self._ep_reward_list,reward)
    self.genes[i] = copy.deepcopy(env_)
    

def mutation(self):
    i1 = np.random.randint(len(self.genes))
    i2 = (i1 + 1 + np.random.randint(len(self.genes) - 1)) % (len(self.genes))
    while True:
        env_ = UAVEnv()
        self.genes[i1] = env_
        self.genes[i1] = self.genes[i2]
        self.genes[i2] = env_
        i1 = (i1 + 1) % len(self.genes)
        i2 = (i2 - 1) % len(self.genes)
        if(i1 == i2 or abs(i1 - i2) == 1):
            break
    update(self)

def mutation1(self):
    x = np.random.randint(len(self.genes),size = 10)
    for i in range(len(x)):
        for j in range(self.genes[x[i]].M):  
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2  
            dis_ue = tmp[1] * self.genes[0].delta_t * self.genes[0].v_ue  
            self.genes[x[i]].loc_ue_list[j][0] = self.genes[x[i]].loc_ue_list[j][0] + math.cos(theta_ue) * dis_ue
            self.genes[x[i]].loc_ue_list[j][1] = self.genes[x[i]].loc_ue_list[j][1] + math.sin(theta_ue) * dis_ue
            self.genes[x[i]].loc_ue_list[j] = np.clip(self.genes[x[i]].loc_ue_list[j], 0, self.genes[x[i]].ground_width)
        self.genes[x[i]].task_list = np.random.randint(2621440, 3145729, self.genes[0].M)
        self.genes[x[i]].block_flag_list = np.random.randint(0, 2, self.genes[0].M)  
    update(self)

def mutation2(self):
    x = np.random.randint(len(self.genes),size = 10)
    for i in range(len(x)):
        tmp = np.random.rand(2)
        theta_uav = tmp[0] * np.pi * 2
        dis_uav = tmp[1] * self.genes[0].flight_speed * self.genes[0].t_fly
        self.genes[x[i]].loc_uav[0] = dis_uav * math.cos(theta_uav) + self.genes[x[i]].loc_uav[0]
        self.genes[x[i]].loc_uav[1] = dis_uav * math.sin(theta_uav) + self.genes[x[i]].loc_uav[1]
        self.genes[x[i]].loc_uav = np.clip(self.genes[x[i]].loc_uav,0,self.genes[0].ground_width)
    update(self)

###############################  training  ####################################


np.random.seed(1)
tf.random.set_seed(1)

env = UAVEnv()
env__ = UAVEnv()
MAX_EP_STEPS = env.slot_num
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound  # [-1,1]
bestReward = []

ddpg = DDPG(a_dim, s_dim, a_bound)

# var = 1  # control exploration
var = 0.01  # control exploration
t1 = time.time()
ep_reward_list = []
s_normal = StateNormalization()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    j = 0
    idv_ = Indv()
    
    while j < MAX_EP_STEPS:
        # Add exploration noise
        if i < MAX_EPISODES - 63:
            a = ddpg.choose_action(s_normal.state_normal(s))
            a = np.clip(np.random.normal(a, var), *a_bound)  # 高斯噪声add randomness to action selection for exploration
            # print(a.shape)
            s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(a)
            if step_redo:
                continue
            if reset_dist:
                a[2] = -1
            if offloading_ratio_change:
                a[3] = -1
            ddpg.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_))  # 训练奖励缩小10倍

            if ddpg.pointer > MEMORY_CAPACITY:
                # var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
                ddpg.learn()
            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1 or is_terminal:
                print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % var)
                ep_reward_list = np.append(ep_reward_list, ep_reward)
                # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
                file_name = 'output_ga_ver1_DDPG.txt'
                with open(file_name, 'a') as file_obj:
                    #file_obj.write("\n======== This episode is done ========")  # 本episode结束
                    print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % var,file = file_obj)
                    # Print and write the episode information to the file with a newline character
                break
            j = j + 1
        if i > MAX_EPISODES - 64:
            a = ddpg.choose_action(s_normal.state_normal(s))
            a = np.clip(np.random.normal(a, var), *a_bound)  # 高斯噪声add randomness to action selection for exploration
            env__ = copy.deepcopy(env)
            # env__.act[0][0] = (a[0] + 1) / 2
            # env__.act[0][1] = (a[1] + 1) / 2
            # env__.act[0][2] = (a[2] + 1) / 2
            # env__.act[0][3] = (a[3] + 1) / 2
            env__.act = (a + 1)/2
            #print(env__.act)
            #print(a)
            # print(env__.act.shape)
            s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(a)
            if step_redo:
                continue
            if reset_dist:
                a[2] = -1
            if offloading_ratio_change:
                a[3] = -1
            ddpg.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_))  # 训练奖励缩小10倍
            env__.sum_task_size = env.sum_task_size
            env__.e_battery_uav = env.e_battery_uav
            idv_._ep_reward += r
            idv_._ep_reward_list = np.append(idv_._ep_reward_list,r)
            idv_.genes = np.append(idv_.genes,env__)
            if ddpg.pointer > MEMORY_CAPACITY:
                # var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
                ddpg.learn()
            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1 or is_terminal:
                print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % var)
                ep_reward_list = np.append(ep_reward_list, ep_reward)
                # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
                file_name = 'output_ga_ver1_DDPG.txt'
                with open(file_name, 'a') as file_obj:
                    #file_obj.write("\n======== This episode is done ========")  # 本episode结束
                    print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Explore: %.3f' % var, file = file_obj)
                    # Print and write the episode information to the file with a newline character
                population = np.append(population,idv_)
                break
            j = j + 1



    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)

# print('Running time: ', time.time() - t1)
# plt.plot(ep_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.show()
###################################figure##############################################
# fig = plt.subplots()
# fig.savefig('ketqua.png')


# def GA():
#     global bestReward
#     # population = []
#     # for i in range(nmbpopu):
#     #     idv = Indv()
#     #     idv._add_state()
#     #     population = np.append(population,idv)
#     for i in range(generations):
#         while(len(population) < populationLimit):
#             i1 = np.random.randint(len(population))
#             i2 = (i1 + 1 + np.random.randint(len(population) - 1)) % len(population)
#             parent1 = population[i1]
#             parent2 = population[i2]
#             if np.random.rand() < 0.3:
#                 child1, child2 = crossOver(parent1,parent2)
#                 if  np.random.rand() < mutation_rate:
#                     mutation(child1)
#                     mutation(child2)
#                 if  np.random.rand() < mutation_rate_1:
#                     mutation1(child1)
#                     mutation1(child2)
#                 if np.random.rand() < mutation_rate_2:
#                     mutation2(child1)
#                     mutation2(child2)
#                 population = np.append(population,child1)
#                 population = np.append(population,child2)
#         sorted_indices = np.argsort([-idv._ep_reward for idv in population])
#         population = population[sorted_indices]

#         population = population[:64]
#         bestState = population[0]
#         #worstState = population[63]
#         bestReward = np.append(bestReward,bestState._ep_reward)
#         print('Generation:', i,' Reward: %7.2f' % bestState._ep_reward, 'mutation_rate: %.3f' % mutation_rate)
#         #print('Generation:', i,' worstReward: %7.2f' % worstState._ep_reward, 'mutation_rate: %.3f' % mutation_rate)

    # population = []
    # for i in range(nmbpopu):
    #     idv = Indv()
    #     idv._add_state()
    #     population = np.append(population,idv)
# for i in range(len(population[0].genes)):
#     print(' E_battery_uav: %7.2f' % population[0].genes[i].e_battery_uav)
for i in range(generations):
    while(len(population) < populationLimit):
        i1 = np.random.randint(len(population))
        i2 = (i1 + 1 + np.random.randint(len(population) - 1)) % len(population)
        parent1 = population[i1]
        parent2 = population[i2]
        if np.random.rand() < 0.3:
            child1, child2 = crossOver(parent1,parent2)
            if  np.random.rand() < mutation_rate:
                mutation(child1)
                mutation(child2)
            if  np.random.rand() < mutation_rate_1:
                mutation1(child1)
                mutation1(child2)
            if np.random.rand() < mutation_rate_2:
                mutation2(child1)
                mutation2(child2)
            population = np.append(population,child1)
            population = np.append(population,child2)
    sorted_indices = np.argsort([-idv._ep_reward for idv in population])
    population = population[sorted_indices]

    population = population[:64]
    bestState = population[0]
    #worstState = population[63]
    bestReward = np.append(bestReward,bestState._ep_reward)
    print('Generation:', i,' Reward: %7.2f' % bestState._ep_reward, 'mutation_rate: %.3f' % mutation_rate)
    file_name = 'output_ga_ver1_GA.txt'
    with open(file_name,'a') as file_obj:
        #print('Generation:', i,' worstReward: %7.2f' % worstState._ep_reward, 'mutation_rate: %.3f' % mutation_rate)
        print('Generation:', i,' Reward: %7.2f' % bestState._ep_reward, 'mutation_rate: %.3f' % mutation_rate, file = file_obj)

print('Running time: ', time.time() - t1)
plt.plot(bestReward)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title('GA_<UE = 4>')
plt.show()
fig = plt.subplots()
##############################################
plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title('DDPG_<UE = 4>')
plt.show()
###################################figure##############################################
fig1 = plt.subplots()





